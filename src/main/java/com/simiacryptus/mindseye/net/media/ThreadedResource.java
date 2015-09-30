package com.simiacryptus.mindseye.net.media;

public abstract class ThreadedResource<T> {

  final int maxItems = 16;
  public java.util.concurrent.LinkedBlockingQueue<T> pool = new java.util.concurrent.LinkedBlockingQueue<>();
  public java.util.HashSet<T> all = new java.util.HashSet<>(maxItems);

  public abstract T create();

  public void with(java.util.function.Consumer<T> f) {
    T poll = pool.poll();
    if (null == poll) {
      synchronized (all) {
        if (all.size() < maxItems) {
          poll = create();
          all.add(poll);
        }
      } 
    }
    if(null==poll){
      try {
        poll = pool.take();
      } catch (InterruptedException e) {
        throw new java.lang.RuntimeException(e);
      }
    }
    f.accept(poll);
    pool.add(poll);
  }
}