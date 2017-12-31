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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.HtmlNotebookOutput;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.CodeUtil;
import com.simiacryptus.util.test.SysOutInterceptor;

import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * The type Notebook output run base.
 */
public abstract class NotebookReportBase {
  
  static {
    SysOutInterceptor.INSTANCE.init();
  }
  
  /**
   * The Use markdown.
   */
  protected boolean useMarkdown = Boolean.parseBoolean(System.getProperty("useMarkdown", "false"));
  /**
   * The Prefer static.
   */
  protected boolean preferStatic = Boolean.parseBoolean(System.getProperty("preferStatic", "true"));
  
  public static String printHeader(NotebookOutput log, Class<?> networkClass, final String prefix) {
    String appJavadoc = CodeUtil.getJavadoc(networkClass);
    log.setFrontMatterProperty(prefix + "_class_short", networkClass.getSimpleName());
    log.setFrontMatterProperty(prefix + "_class_full", networkClass.getCanonicalName());
    log.setFrontMatterProperty(prefix + "_class_doc", appJavadoc.replaceAll("\n", ""));
    return appJavadoc;
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  public void run() {
    try (NotebookOutput log = getLog()) {
      printHeader(log);
      run(log);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  protected abstract void run(NotebookOutput log);
  
  public void printHeader(NotebookOutput log) {
    log.setFrontMatterProperty("created_on", new Date().toString());
    String targetJavadoc = printHeader(log, getTargetClass(), "network");
    String reportJavadoc = printHeader(log, getReportClass(), "report");
    log.p("__Target Description:__ " + targetJavadoc);
    log.p("__Report Description:__ " + reportJavadoc);
  }
  
  public Class<? extends NotebookReportBase> getReportClass() {
    return getClass();
  }
  
  /**
   * Gets log.
   *
   * @return the log
   */
  public NotebookOutput getLog() {
    try {
      if (useMarkdown) {
        String absoluteUrl = "https://github.com/SimiaCryptus/MindsEye/tree/master/src/";
        return MarkdownNotebookOutput.get(getTargetClass(),
                                          absoluteUrl,
                                          getClass().getSimpleName());
      }
      else {
        final String directoryName = new SimpleDateFormat("YYYY-MM-dd-HH-mm").format(new Date());
        final File path = new File(Util.mkString(File.separator, "www", directoryName));
        path.mkdirs();
        final File logFile = new File(path, "index.html");
        final HtmlNotebookOutput log;
        if (preferStatic) {
          log = new HtmlNotebookOutput(path, new FileOutputStream(logFile));
          Desktop.getDesktop().browse(logFile.toURI());
        }
        else {
          final StreamNanoHTTPD server = new StreamNanoHTTPD(1999, "text/html", logFile).init();
          log = new HtmlNotebookOutput(path, server.dataReciever);
        }
        return log;
      }
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  protected abstract Class<?> getTargetClass();
}
