/*
 * Copyright (c) 2017 by Andrew Charneski.
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

package com.simiacryptus.util.test;

import com.simiacryptus.util.lang.UncheckedSupplier;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

/**
 * The type Sys out interceptor.
 */
public class SysOutInterceptor extends PrintStream {
  
  /**
   * The constant INSTANCE.
   */
  public static final SysOutInterceptor INSTANCE = init();
  private final ThreadLocal<PrintStream> threadHandler = new ThreadLocal<PrintStream>() {
    @Override
    protected PrintStream initialValue() {
      return getInner();
    }
  };
  private final ThreadLocal<Boolean> isMonitoring = new ThreadLocal<Boolean>() {
    @Override
    protected Boolean initialValue() {
      return false;
    }
  };
  
  /**
   * Instantiates a new Sys out interceptor.
   *
   * @param out the out
   */
  private SysOutInterceptor(PrintStream out) {
    super(out);
  }
  
  /**
   * With output logged result.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the logged result
   */
  public static <T> LoggedResult<T> withOutput(UncheckedSupplier<T> fn) {
    //init();
    PrintStream prev = INSTANCE.threadHandler.get();
    try {
      ByteArrayOutputStream buff = new ByteArrayOutputStream();
      try (PrintStream ps = new PrintStream(buff)) {
        INSTANCE.threadHandler.set(ps);
        T result = fn.get();
        ps.close();
        return new LoggedResult<T>(result, buff.toString());
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    } finally {
      INSTANCE.threadHandler.set(prev);
    }
  }
  
  /**
   * With output logged result.
   *
   * @param fn the fn
   * @return the logged result
   */
  public static LoggedResult<Void> withOutput(Runnable fn) {
    try {
      ByteArrayOutputStream buff = new ByteArrayOutputStream();
      try (PrintStream ps = new PrintStream(buff)) {
        if (INSTANCE.isMonitoring.get()) throw new IllegalStateException();
        INSTANCE.threadHandler.set(ps);
        INSTANCE.isMonitoring.set(true);
        fn.run();
        return new LoggedResult<Void>(null, buff.toString());
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    } finally {
      INSTANCE.threadHandler.remove();
      INSTANCE.isMonitoring.remove();
    }
  }
  
  private static SysOutInterceptor init() {
    if (!(System.out instanceof SysOutInterceptor)) {
      SysOutInterceptor out = new SysOutInterceptor(System.out);
      System.setOut(out);
      return out;
    }
    return (SysOutInterceptor) System.out;
  }
  
  /**
   * Gets inner.
   *
   * @return the inner
   */
  public PrintStream getInner() {
    return (PrintStream) out;
  }
  
  @Override
  public void print(String s) {
    currentHandler().print(s);
  }
  
  @Override
  public void println(String x) {
    PrintStream currentHandler = currentHandler();
    currentHandler.println(x);
  }
  
  /**
   * Current handler print stream.
   *
   * @return the print stream
   */
  public PrintStream currentHandler() {
    return threadHandler.get();
  }
  
  /**
   * Sets current handler.
   *
   * @param out the out
   */
  public void setCurrentHandler(PrintStream out) {
    threadHandler.set(out);
  }
  
  /**
   * The type Logged result.
   *
   * @param <T> the type parameter
   */
  public static class LoggedResult<T> {
    /**
     * The Obj.
     */
    public final T obj;
    /**
     * The Log.
     */
    public final String log;

    /**
     * Instantiates a new Logged result.
     *
     * @param obj the obj
     * @param log the log
     */
    public LoggedResult(T obj, String log) {
      this.obj = obj;
      this.log = log;
    }
  }
}
