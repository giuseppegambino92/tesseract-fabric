//! Persistent intra-node lane pool: `lanes − 1` helper threads plus the
//! caller execute the tasks of one parallel region. Helpers spin briefly on a
//! generation counter before parking, so back-to-back regions (one per GEMM
//! phase of a layer) pay no OS thread spawn and usually no wake-up.
//!
//! Task claiming uses a single `AtomicU64` holding (generation << 32 | next
//! index): a helper that was slow to notice a region can never claim a task
//! of a later one, and the caller returns only after every claimed task of
//! its region has finished, so the borrowed closure outlives all its calls.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    /// (generation, job, number of tasks, timed) of the current region.
    region: Mutex<(u64, Option<JobPtr>, usize, bool)>,
    wake: Condvar,
    gen: AtomicU64,
    next: AtomicU64,
    done: AtomicU64,
    stop: AtomicBool,
    /// Timed regions: the largest CPU time any one lane spent on tasks.
    max_cpu: AtomicU64,
}

#[derive(Clone, Copy)]
struct JobPtr(Job);
// SAFETY: the pointee is `Sync` and outlives every use (see module docs).
unsafe impl Send for JobPtr {}

pub struct LanePool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    lanes: usize,
}

const SPIN_NS: u64 = 50_000;

impl LanePool {
    pub fn new(lanes: usize) -> Self {
        let lanes = lanes.max(1);
        let shared = Arc::new(Shared {
            region: Mutex::new((0, None, 0, false)),
            wake: Condvar::new(),
            gen: AtomicU64::new(0),
            next: AtomicU64::new(0),
            done: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            max_cpu: AtomicU64::new(0),
        });
        let workers = (1..lanes)
            .map(|_| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name("tess-lane".into())
                    .spawn(move || worker(sh))
                    .expect("spawn lane")
            })
            .collect();
        LanePool {
            shared,
            workers,
            lanes,
        }
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Run `f(0..n)` across the lanes; returns when every task has finished.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if self.lanes == 1 || n == 1 {
            (0..n).for_each(f);
            return;
        }
        let sh = &self.shared;
        // SAFETY: `f` is only called for tasks of this generation, and this
        // function does not return before all of them have completed.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f) };
        // inside a critical-path measurement, account the region by its
        // busiest lane instead of the caller's own (spinning) CPU time
        let cpu0 = if TIMED.with(|t| t.get()) {
            thread_cpu_ns()
        } else {
            None
        };
        let timed = cpu0.is_some();
        sh.max_cpu.store(0, Ordering::SeqCst);
        let g = sh.gen.load(Ordering::Relaxed) + 1;
        sh.next.store(g << 32, Ordering::SeqCst);
        sh.done.store(g << 32, Ordering::SeqCst);
        {
            let mut r = sh.region.lock().unwrap();
            *r = (g, Some(JobPtr(job)), n, timed);
            sh.gen.store(g, Ordering::SeqCst);
        }
        sh.wake.notify_all();
        IN_REGION.with(|r| r.set(true));
        claim(sh, g, n, job, timed);
        IN_REGION.with(|r| r.set(false));
        let want = (g << 32) | n as u64;
        while sh.done.load(Ordering::Acquire) != want {
            std::hint::spin_loop();
        }
        sh.region.lock().unwrap().1 = None;
        if let (Some(c0), Some(c1)) = (cpu0, thread_cpu_ns()) {
            let lane = sh.max_cpu.load(Ordering::Acquire) as i64;
            CRIT_ADJ.with(|a| a.set(a.get() + lane - c1.saturating_sub(c0) as i64));
        }
    }
}

fn claim(sh: &Shared, g: u64, n: usize, job: Job, timed: bool) {
    // timed regions: this lane's CPU time on tasks, published before each
    // task is marked done so the caller sees every lane's total
    let mut spent = 0u64;
    let mut mark = if timed {
        thread_cpu_ns().unwrap_or(0)
    } else {
        0
    };
    loop {
        let cur = sh.next.load(Ordering::Acquire);
        if cur >> 32 != g || (cur & 0xFFFF_FFFF) as usize >= n {
            return;
        }
        if sh
            .next
            .compare_exchange(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            job((cur & 0xFFFF_FFFF) as usize);
            if timed {
                let now = thread_cpu_ns().unwrap_or(mark);
                spent += now.saturating_sub(mark);
                mark = now;
                sh.max_cpu.fetch_max(spent, Ordering::AcqRel);
            }
            sh.done.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn worker(sh: Arc<Shared>) {
    crate::tensor::prefer_performance_cores();
    let mut seen = 0u64;
    loop {
        // spin, then park, until a new generation (or stop) appears
        let t0 = crate::wire::transport::mono_ns();
        while sh.gen.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Acquire) {
            if crate::wire::transport::mono_ns() - t0 > SPIN_NS {
                let r = sh.region.lock().unwrap();
                let _r = sh
                    .wake
                    .wait_while(r, |r| r.0 == seen && !sh.stop.load(Ordering::Acquire))
                    .unwrap();
                break;
            }
            std::hint::spin_loop();
        }
        if sh.stop.load(Ordering::Acquire) {
            return;
        }
        let (g, job, n, timed) = {
            let r = sh.region.lock().unwrap();
            (r.0, r.1, r.2, r.3)
        };
        seen = g;
        if let Some(JobPtr(job)) = job {
            claim(&sh, g, n, job, timed);
        }
    }
}

impl Drop for LanePool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        {
            let _r = self.shared.region.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<std::rc::Rc<LanePool>>> =
        const { std::cell::RefCell::new(None) };
    /// Set while this thread is inside a parallel region (no nesting).
    static IN_REGION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set while a [`critical_path`] measurement is running on this thread.
    static TIMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Correction from this thread's CPU time to critical-path time: busiest
    /// lane minus the caller's own CPU per region, plus [`count_wall`] waits.
    static CRIT_ADJ: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// CPU time consumed so far by the calling thread, in ns (`None` where the
/// OS has no per-thread clock). Unlike wall time it does not advance while
/// the thread is runnable but descheduled by other load on the host.
pub fn thread_cpu_ns() -> Option<u64> {
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android"
    ))]
    {
        use std::os::raw::{c_int, c_long};
        #[repr(C)]
        struct Timespec {
            sec: c_long,
            nsec: c_long,
        }
        extern "C" {
            fn clock_gettime(clk: c_int, ts: *mut Timespec) -> c_int;
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const CLOCK_THREAD_CPUTIME_ID: c_int = 16;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        const CLOCK_THREAD_CPUTIME_ID: c_int = 3;
        let mut ts = Timespec { sec: 0, nsec: 0 };
        // SAFETY: plain libc call writing into a valid, owned timespec.
        if unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) } == 0 {
            return Some(ts.sec as u64 * 1_000_000_000 + ts.nsec as u64);
        }
        None
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android"
    )))]
    {
        None
    }
}

/// Run `f` and return `(output, wall ns, critical-path ns)`.
///
/// The critical path is the compute that actually ran: the caller's CPU
/// time, with each parallel region counted by its busiest lane, plus any
/// [`count_wall`] waits. Time the host spent running *other* work on these
/// threads (daemons, other emulated nodes) is excluded. It is clamped to the
/// wall time, and equals it on platforms without a thread clock.
pub fn critical_path<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let now = crate::wire::transport::mono_ns;
    let Some(c0) = thread_cpu_ns() else {
        let t0 = now();
        let out = f();
        let dt = now() - t0;
        return (out, dt, dt);
    };
    let prev = TIMED.with(|t| t.replace(true));
    let a0 = CRIT_ADJ.with(|a| a.get());
    let t0 = now();
    let out = f();
    let dt = now() - t0;
    let c1 = thread_cpu_ns().unwrap_or(c0);
    let adj = CRIT_ADJ.with(|a| a.get()) - a0;
    TIMED.with(|t| t.set(prev));
    let crit = (c1.saturating_sub(c0) as i64 + adj).clamp(0, dt as i64) as u64;
    (out, dt, crit)
}

/// Run `f`, a genuine wait (on another node or thread), and count its full
/// wall time on the critical path of an enclosing [`critical_path`].
pub fn count_wall<T>(f: impl FnOnce() -> T) -> T {
    if !TIMED.with(|t| t.get()) {
        return f();
    }
    let now = crate::wire::transport::mono_ns;
    let (t0, c0) = (now(), thread_cpu_ns().unwrap_or(0));
    let out = f();
    let wall = now() - t0;
    let cpu = thread_cpu_ns().unwrap_or(c0).saturating_sub(c0);
    CRIT_ADJ.with(|a| a.set(a.get() + wall as i64 - cpu as i64));
    out
}

/// Give the calling node thread `lanes` lanes (1 = none): every engine's
/// node threads install the same count, so both sides of every comparison
/// run on the same per-node hardware.
pub fn install(lanes: usize) {
    CURRENT.with(|c| {
        *c.borrow_mut() = (lanes > 1).then(|| std::rc::Rc::new(LanePool::new(lanes)));
    });
}

/// The calling thread's lane pool, if any (none inside a parallel region).
pub fn current() -> Option<std::rc::Rc<LanePool>> {
    if IN_REGION.with(|r| r.get()) {
        return None;
    }
    CURRENT.with(|c| c.borrow().clone())
}

/// Raw pointer that may be shared across lanes for writes to disjoint ranges.
#[derive(Clone, Copy)]
pub struct SyncPtr<T = f32>(pub *mut T);
// SAFETY: users only write disjoint index ranges from different tasks.
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// The raw pointer (a method, so closures capture the `Sync` wrapper).
    pub fn ptr(&self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn pool_runs_every_task_exactly_once_across_many_regions() {
        let pool = LanePool::new(4);
        for round in 0..500usize {
            let n = 1 + round % 37;
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    /// Burn `ns` of this thread's CPU time.
    fn burn(ns: u64) {
        let c0 = thread_cpu_ns().unwrap();
        while thread_cpu_ns().unwrap() - c0 < ns {
            std::hint::black_box(0u64);
        }
    }

    #[test]
    fn critical_path_counts_compute_and_marked_waits_but_not_idle_time() {
        if thread_cpu_ns().is_none() {
            return;
        }
        let ms = 1_000_000u64;
        // CPU-bound work is counted
        let (_, wall, crit) = critical_path(|| burn(8 * ms));
        assert!(crit >= 8 * ms && crit <= wall, "{crit} {wall}");
        // time off-CPU (a stand-in for being descheduled) is not
        let (_, wall, crit) =
            critical_path(|| std::thread::sleep(std::time::Duration::from_millis(20)));
        assert!(wall >= 20 * ms && crit < 5 * ms, "{crit} {wall}");
        // ...unless it is a genuine wait marked with count_wall
        let (_, wall, crit) = critical_path(|| {
            count_wall(|| std::thread::sleep(std::time::Duration::from_millis(20)))
        });
        assert!(crit >= 20 * ms && crit <= wall, "{crit} {wall}");
        // a parallel region counts its busiest lane, not the caller's spin
        let pool = LanePool::new(4);
        let (_, wall, crit) = critical_path(|| {
            pool.run(4, &|i| {
                if i == 3 {
                    burn(12 * ms)
                } else {
                    burn(2 * ms)
                }
            })
        });
        assert!(crit >= 12 * ms && crit <= wall, "{crit} {wall}");
    }
}
