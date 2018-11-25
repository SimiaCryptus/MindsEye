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

import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
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
   * Gets test report location.
   *
   * @param sourceClass the source class
   * @param reportingFolder
   * @param suffix      the suffix
   * @return the test report location
   */
  @Nonnull
  public static File getTestReportLocation(@Nonnull final Class<?> sourceClass, String reportingFolder, @Nonnull final CharSequence... suffix) {
    final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
    final CharSequence methodName = callingFrame.getMethodName();
    final String className = sourceClass.getCanonicalName();
    String classFilename = className.replaceAll("\\.", "/").replaceAll("\\$", "/");
    @Nonnull File path = new File(Util.mkString(File.separator, reportingFolder, classFilename));
    for (int i = 0; i < suffix.length - 1; i++) path = new File(path, suffix[i].toString());
    String testName = suffix.length == 0 ? String.valueOf(methodName) : suffix[suffix.length - 1].toString();
    File parent = path;
    //parent = new File(path, new SimpleDateFormat("yyyy-MM-dd_HHmmss").format(new Date()));
    path = new File(parent, testName + ".md");
    path.getParentFile().mkdirs();
    logger.info(String.format("Output Location: %s", path.getAbsoluteFile()));
    return path;
  }

  /**
   * Gets log.
   *
   * @param path the report location
   * @return the log
   */
  @Nonnull
  public static NotebookOutput getLog(final File path) {
    try {
      StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
      String methodName = callingFrame.getMethodName();
      path.getParentFile().mkdirs();
      return new MarkdownNotebookOutput(new File(path, methodName), TestSettings.INSTANCE.autobrowse);
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Gets report type.
   *
   * @return the report type
   */
  @Nonnull
  public abstract ReportType getReportType();

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
   * Run.
   *
   * @param fn      the fn
   * @param logPath the log path
   */
  public void run(@Nonnull Consumer<NotebookOutput> fn, @Nonnull CharSequence... logPath) {
    try (@Nonnull NotebookOutput log = getLog(logPath)) {
      NotebookOutput.concat(this::printHeader, MarkdownNotebookOutput.wrapFrontmatter(fn)).accept(log);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Gets log.
   *
   * @param logPath the log path
   * @return the log
   */
  @Nonnull
  public NotebookOutput getLog(CharSequence... logPath) {
    if (null == logPath || logPath.length == 0) logPath = new String[]{getClass().getSimpleName()};
    return getLog(getTestReportLocation(getTargetClass(), reportingFolder, logPath));
  }
  protected String reportingFolder = "reports/_reports";

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

}
