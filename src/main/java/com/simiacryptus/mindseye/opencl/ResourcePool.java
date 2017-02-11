package com.simiacryptus.mindseye.opencl;

public abstract class ResourcePool<T> {

  private final java.util.HashSet<T> all;
  private final java.util.concurrent.LinkedBlockingQueue<T> pool = new java.util.concurrent.LinkedBlockingQueue<>();
  private final int maxItems;
  
  public ResourcePool(int maxItems) {
    super();
    this.maxItems = maxItems;
    this.all = new java.util.HashSet<>(this.maxItems);
  }

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
