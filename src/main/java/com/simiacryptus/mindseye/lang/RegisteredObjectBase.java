/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.lang;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.*;
import java.util.stream.Stream;

/**
 * The type Registered object base.
 */
public abstract class RegisteredObjectBase extends ReferenceCountingBase {
  private static final Logger logger = LoggerFactory.getLogger(RegisteredObjectBase.class);
  private static final Map<Class<? extends RegisteredObjectBase>, ObjectRecords<RegisteredObjectBase>> cache = new ConcurrentHashMap<>();
  private static final ScheduledExecutorService maintenanceThread = Executors.newScheduledThreadPool(1, new ThreadFactoryBuilder().setDaemon(true).build());

  /**
   * Instantiates a new Registered object base.
   */
  public RegisteredObjectBase() {
    cache.computeIfAbsent(getClass(), k -> new ObjectRecords<>()).add(new WeakReference<>(this));
  }

  /**
   * Gets living instances.
   *
   * @param <T> the type parameter
   * @param k   the k
   * @return the living instances
   */
  public static <T extends RegisteredObjectBase> Stream<T> getLivingInstances(final Class<T> k) {
    return getInstances(k).filter(x -> !x.isFinalized());
  }

  /**
   * Gets instances.
   *
   * @param <T> the type parameter
   * @param k   the k
   * @return the instances
   */
  public static <T extends RegisteredObjectBase> Stream<T> getInstances(final Class<T> k) {
    return cache.entrySet().stream().filter(e -> k.isAssignableFrom(e.getKey())).map(x -> x.getValue())
        .flatMap(ObjectRecords::stream).map(x -> (T) x.get()).filter(x -> x != null);
  }

  private static class ObjectRecords<T extends RegisteredObjectBase> extends ConcurrentLinkedDeque<WeakReference<T>> {
    private volatile boolean dirty = false;
    private final ScheduledFuture<?> maintenanceFuture = maintenanceThread.scheduleAtFixedRate(
        this::maintain, 1, 1, TimeUnit.SECONDS);

    private void maintain() {
      if (dirty) {
        this.removeIf(ref -> null == ref.get());
        dirty = false;
      }
    }

    @Override
    public boolean add(final WeakReference<T> tWeakReference) {
      dirty = true;
      return super.add(tWeakReference);
    }

    @Override
    public Stream<WeakReference<T>> stream() {
      dirty = true;
      return super.stream();
    }
  }

}
