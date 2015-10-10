package com.simiacryptus.mindseye.net.media;

public abstract class ThreadedResource<T> {

  public java.util.HashSet<T> all = new java.util.HashSet<>(this.maxItems);
  final int maxItems = 16;
  public java.util.concurrent.LinkedBlockingQueue<T> pool = new java.util.concurrent.LinkedBlockingQueue<>();

  public abstract T create();

  public void with(final java.util.function.Consumer<T> f) {
    T poll = this.pool.poll();
    if (null == poll) {
      synchronized (this.all) {
        if (this.all.size() < this.maxItems) {
          poll = create();
          this.all.add(poll);
        }
      }
    }
    if (null == poll) {
      try {
        poll = this.pool.take();
      } catch (final InterruptedException e) {
        throw new java.lang.RuntimeException(e);
      }
    }
    f.accept(poll);
    this.pool.add(poll);
  }
}
