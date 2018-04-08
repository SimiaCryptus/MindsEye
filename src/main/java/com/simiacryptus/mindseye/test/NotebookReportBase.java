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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.HtmlNotebookOutput;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.CodeUtil;
import com.simiacryptus.util.lang.TimedResult;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.function.Consumer;

/**
 * The type Notebook output apply base.
 */
public abstract class NotebookReportBase {
  
  /**
   * The constant log.
   */
  protected static final Logger logger = LoggerFactory.getLogger(NotebookReportBase.class);
  
  static {
    SysOutInterceptor.INSTANCE.init();
  }
  
  /**
   * The Use markdown.
   */
  protected boolean useMarkdown = Boolean.parseBoolean(System.getProperty("useMarkdown", "true"));
  /**
   * The Prefer static.
   */
  protected boolean preferStatic = Boolean.parseBoolean(System.getProperty("preferStatic", "true"));
  
  private String tag = System.getProperty("GIT_TAG","master");
  /**
   * The Absolute url.
   */
  @Nonnull
  protected CharSequence absoluteUrl = "https://github.com/SimiaCryptus/MindsEye/tree/" + tag + "/src/";
  
  /**
   * Print header string.
   *
   * @param log          the log
   * @param networkClass the network class
   * @param prefix       the prefix
   * @return the string
   */
  @Nullable
  public static CharSequence printHeader(@Nonnull NotebookOutput log, @Nullable Class<?> networkClass, final CharSequence prefix) {
    if (null == networkClass) return null;
    @Nullable String javadoc = CodeUtil.getJavadoc(networkClass);
    log.setFrontMatterProperty(prefix + "_class_short", networkClass.getSimpleName());
    log.setFrontMatterProperty(prefix + "_class_full", networkClass.getCanonicalName());
    log.setFrontMatterProperty(prefix + "_class_doc", javadoc.replaceAll("\n", ""));
    return javadoc;
  }
  
  /**
   * Gets report type.
   *
   * @return the report type
   */
  @Nonnull
  public abstract ReportType getReportType();
  
  /**
   * Run.
   *
   * @param fn      the fn
   * @param logPath the log path
   */
  public void run(@Nonnull Consumer<NotebookOutput> fn, @Nonnull CharSequence... logPath) {
    try (@Nonnull NotebookOutput log = getLog(logPath.length == 0 ? new String[]{getClass().getSimpleName()} : logPath)) {
      printHeader(log);
      @Nonnull TimedResult<Void> time = TimedResult.time(() -> {
        try {
          fn.accept(log);
          log.setFrontMatterProperty("result", "OK");
        } catch (Throwable e) {
          log.setFrontMatterProperty("result", getExceptionString(e).toString().replaceAll("\n", "<br/>").trim());
          throw (RuntimeException) (e instanceof RuntimeException ? e : new RuntimeException(e));
        }
      });
      log.setFrontMatterProperty("execution_time", String.format("%.6f", time.timeNanos / 1e9));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Nonnull
  private CharSequence getExceptionString(Throwable e) {
    if (e instanceof RuntimeException && e.getCause() != null && e.getCause() != e)
      return getExceptionString(e.getCause());
    if (e.getCause() != null && e.getCause() != e)
      return e.getClass().getSimpleName() + " / " + getExceptionString(e.getCause());
    return e.getClass().getSimpleName();
  }
  
  /**
   * Print header.
   *
   * @param log the log
   */
  public void printHeader(@Nonnull NotebookOutput log) {
    log.setFrontMatterProperty("created_on", new Date().toString());
    log.setFrontMatterProperty("report_type", getReportType().name());
    @Nullable CharSequence targetJavadoc = printHeader(log, getTargetClass(), "network");
    @Nullable CharSequence reportJavadoc = printHeader(log, getReportClass(), "report");
//    log.p("__Target Description:__ " + StringEscapeUtils.escapeHtml4(targetJavadoc));
//    log.p("__Report Description:__ " + StringEscapeUtils.escapeHtml4(reportJavadoc));
    log.p("__Target Description:__ " + targetJavadoc);
    log.p("__Report Description:__ " + reportJavadoc);
  }
  
  /**
   * Gets report class.
   *
   * @return the report class
   */
  public Class<? extends NotebookReportBase> getReportClass() {
    return getClass();
  }
  
  /**
   * Gets log.
   *
   * @param logPath the log path
   * @return the log
   */
  @Nonnull
  public NotebookOutput getLog(CharSequence... logPath) {
    try {
      if (useMarkdown) {
        return MarkdownNotebookOutput.get(getTargetClass(), absoluteUrl, logPath);
      }
      else {
        @Nonnull final CharSequence directoryName = new SimpleDateFormat("YYYY-MM-dd-HH-mm").format(new Date());
        @Nonnull final File path = new File(Util.mkString(File.separator, "www", directoryName));
        path.mkdirs();
        @Nonnull final File logFile = new File(path, "index.html");
        @Nonnull final HtmlNotebookOutput log;
        if (preferStatic) {
          log = new HtmlNotebookOutput(path, new FileOutputStream(logFile));
          Desktop.getDesktop().browse(logFile.toURI());
        }
        else {
          @Nonnull final StreamNanoHTTPD server = new StreamNanoHTTPD(1999, "text/html", logFile).init();
          log = new HtmlNotebookOutput(path, server.dataReciever);
        }
        return log;
      }
    } catch (@Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  protected abstract Class<?> getTargetClass();
  
  /**
   * The enum Report type.
   */
  public enum ReportType {
    /**
     * Demos report type.
     */
    Applications,
    /**
     * Components report type.
     */
    Components,
    /**
     * Models report type.
     */
    Models,
    /**
     * Data report type.
     */
    Data,
    /**
     * Optimizers report type.
     */
    Optimizers, /**
     * Training report type.
     */
    Experiments
  }
  
  /**
   * The type Simple notebook report base.
   */
  public abstract static class SimpleNotebookReportBase extends NotebookReportBase {
    /**
     * Run.
     */
    public void run() {
      run(this::run);
    }
  
    /**
     * Run.
     *
     * @param notebookOutput the notebook output
     */
    protected abstract void run(NotebookOutput notebookOutput);
  }
}
