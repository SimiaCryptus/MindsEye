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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The type Sys out interceptor.
 */
public class SysOutInterceptor extends PrintStream {
  
  /**
   * The constant INSTANCE.
   */
  public static final PrintStream ORIGINAL_OUT = System.out;
  private static final Logger log = LoggerFactory.getLogger(SysOutInterceptor.class);
  /**
   * The constant INSTANCE.
   */
  public static final SysOutInterceptor INSTANCE = new SysOutInterceptor(ORIGINAL_OUT).init();
  private final ThreadLocal<Boolean> isMonitoring = new ThreadLocal<Boolean>() {
    @Override
    protected Boolean initialValue() {
      return false;
    }
  };
  private final ThreadLocal<PrintStream> threadHandler = new ThreadLocal<PrintStream>() {
    @Override
    protected PrintStream initialValue() {
      return getInner();
    }
  };
  
  private final AtomicBoolean initialized = new AtomicBoolean(false);
  
  /**
   * Instantiates a new Sys out interceptor.
   *
   * @param out the out
   */
  private SysOutInterceptor(final PrintStream out) {
    super(out);
  }
  
  /**
   * With output logged result.
   *
   * @param fn the fn
   * @return the logged result
   */
  public static LoggedResult<Void> withOutput(final Runnable fn) {
    try {
      final ByteArrayOutputStream buff = new ByteArrayOutputStream();
      try (PrintStream ps = new PrintStream(buff)) {
        if (SysOutInterceptor.INSTANCE.isMonitoring.get()) throw new IllegalStateException();
        SysOutInterceptor.INSTANCE.threadHandler.set(ps);
        SysOutInterceptor.INSTANCE.isMonitoring.set(true);
        fn.run();
        return new LoggedResult<>(null, buff.toString());
      }
    } catch (final Exception e) {
      throw new RuntimeException(e);
    } finally {
      SysOutInterceptor.INSTANCE.threadHandler.remove();
      SysOutInterceptor.INSTANCE.isMonitoring.remove();
    }
  }
  
  /**
   * With output logged result.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the logged result
   */
  public static <T> LoggedResult<T> withOutput(final UncheckedSupplier<T> fn) {
    //init();
    final PrintStream prev = SysOutInterceptor.INSTANCE.threadHandler.get();
    try {
      final ByteArrayOutputStream buff = new ByteArrayOutputStream();
      try (PrintStream ps = new PrintStream(buff)) {
        SysOutInterceptor.INSTANCE.threadHandler.set(ps);
        final T result = fn.get();
        ps.close();
        return new LoggedResult<>(result, buff.toString());
      }
    } catch (final Exception e) {
      throw new RuntimeException(e);
    } finally {
      SysOutInterceptor.INSTANCE.threadHandler.set(prev);
    }
  }
  
  /**
   * Init sys out interceptor.
   *
   * @return the sys out interceptor
   */
  public SysOutInterceptor init() {
    if (!initialized.getAndSet(true)) {
      ch.qos.logback.classic.Logger root = ((ch.qos.logback.classic.Logger) log).getLoggerContext().getLogger("ROOT");
      ch.qos.logback.core.ConsoleAppender stdout = (ch.qos.logback.core.ConsoleAppender) root.getAppender("STDOUT");
      stdout.setOutputStream(this);
      System.setOut(this);
    }
    return this;
  }
  
  /**
   * Current handler printGroups stream.
   *
   * @return the printGroups stream
   */
  public PrintStream currentHandler() {
    return threadHandler.get();
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
  public void print(final String s) {
    currentHandler().print(s);
  }
  
  @Override
  public void write(byte[] b) throws IOException {
    currentHandler().print(new String(b));
  }
  
  @Override
  public void println(final String x) {
    final PrintStream currentHandler = currentHandler();
    currentHandler.println(x);
  }
  
  /**
   * Sets current handler.
   *
   * @param out the out
   */
  public void setCurrentHandler(final PrintStream out) {
    threadHandler.set(out);
  }
  
  /**
   * The type Logged result.
   *
   * @param <T> the type parameter
   */
  public static class LoggedResult<T> {
    /**
     * The Log.
     */
    public final String log;
    /**
     * The Obj.
     */
    public final T obj;
  
    /**
     * Instantiates a new Logged result.
     *
     * @param obj the obj
     * @param log the log
     */
    public LoggedResult(final T obj, final String log) {
      this.obj = obj;
      this.log = log;
    }
  }
}
