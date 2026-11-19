//! 单生产者 / 单消费者的无锁环形缓冲（f32）：输入回调往里写，输出回调往外读，两边都不会阻塞。

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Inner {
    buf: UnsafeCell<Box<[f32]>>,
    mask: usize,
    head: AtomicUsize, // 已写入的总数
    tail: AtomicUsize, // 已读出的总数
}

// 只有 Producer 写 buf 的"空闲区"、只有 Consumer 读"已写区"，两者由 head / tail 隔开
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

pub struct Producer(Arc<Inner>);
pub struct Consumer(Arc<Inner>);

/// 容量向上取到 2 的幂。
pub fn channel(capacity: usize) -> (Producer, Consumer) {
    let cap = capacity.next_power_of_two().max(64);
    let inner = Arc::new(Inner { buf: UnsafeCell::new(vec![0.0; cap].into_boxed_slice()), mask: cap - 1, head: AtomicUsize::new(0), tail: AtomicUsize::new(0) });
    (Producer(inner.clone()), Consumer(inner))
}

impl Producer {
    /// 写入尽可能多的样本，返回实际写入的个数（满了就丢掉多余的）。
    pub fn push_slice(&mut self, data: &[f32]) -> usize {
        let i = &*self.0;
        let head = i.head.load(Ordering::Relaxed);
        let tail = i.tail.load(Ordering::Acquire);
        let free = i.mask + 1 - head.wrapping_sub(tail);
        let n = data.len().min(free);
        let buf = unsafe { &mut *i.buf.get() };
        for (k, &v) in data[..n].iter().enumerate() {
            buf[(head.wrapping_add(k)) & i.mask] = v;
        }
        i.head.store(head.wrapping_add(n), Ordering::Release);
        n
    }
}

impl Consumer {
    pub fn len(&self) -> usize {
        let i = &*self.0;
        i.head.load(Ordering::Acquire).wrapping_sub(i.tail.load(Ordering::Relaxed))
    }

    /// 读出尽可能多的样本，返回实际读出的个数。
    pub fn pop_slice(&mut self, out: &mut [f32]) -> usize {
        let i = &*self.0;
        let tail = i.tail.load(Ordering::Relaxed);
        let avail = i.head.load(Ordering::Acquire).wrapping_sub(tail);
        let n = out.len().min(avail);
        let buf = unsafe { &*i.buf.get() };
        for (k, o) in out[..n].iter_mut().enumerate() {
            *o = buf[(tail.wrapping_add(k)) & i.mask];
        }
        i.tail.store(tail.wrapping_add(n), Ordering::Release);
        n
    }

    /// 丢掉最旧的 n 个样本（积压太多时用来追上实时）。
    pub fn skip(&mut self, n: usize) {
        let i = &*self.0;
        let tail = i.tail.load(Ordering::Relaxed);
        let avail = i.head.load(Ordering::Acquire).wrapping_sub(tail);
        i.tail.store(tail.wrapping_add(n.min(avail)), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_passes_data_between_threads_in_order() {
        let (mut p, mut c) = channel(1024);
        let writer = std::thread::spawn(move || {
            let mut next = 0u32;
            while next < 200_000 {
                let chunk: Vec<f32> = (next..(next + 100).min(200_000)).map(|v| v as f32).collect();
                let mut off = 0;
                while off < chunk.len() {
                    off += p.push_slice(&chunk[off..]);
                    std::thread::yield_now();
                }
                next += chunk.len() as u32;
            }
        });
        let mut expect = 0u32;
        let mut buf = [0f32; 77];
        while expect < 200_000 {
            let n = c.pop_slice(&mut buf);
            for &v in &buf[..n] {
                assert_eq!(v, expect as f32);
                expect += 1;
            }
        }
        writer.join().unwrap();
        assert_eq!(c.len(), 0);
    }
}
