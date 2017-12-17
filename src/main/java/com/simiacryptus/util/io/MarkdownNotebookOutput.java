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

package com.simiacryptus.util.io;

import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.lang.CodeUtil;
import com.simiacryptus.util.lang.TimedResult;
import com.simiacryptus.util.lang.UncheckedSupplier;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.apache.commons.io.IOUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * The type Markdown notebook output.
 */
public class MarkdownNotebookOutput implements NotebookOutput {
  
  private static int excerptNumber = 0;
  private static int imageNumber = 0;
  private final File fileName;
  private final String name;
  private final List<PrintStream> outs = new ArrayList<>();
  private final OutputStream primaryOut;
  /**
   * The Anchor.
   */
  int anchor = 0;
  private String absoluteUrl = null;
  
  /**
   * Instantiates a new Markdown notebook output.
   *
   * @param fileName the file name
   * @param name     the name
   * @throws FileNotFoundException the file not found exception
   */
  public MarkdownNotebookOutput(final File fileName, final String name) throws FileNotFoundException {
    this.name = name;
    primaryOut = new FileOutputStream(fileName);
    outs.add(new PrintStream(primaryOut));
    this.fileName = fileName;
  }
  
  /**
   * Get markdown notebook output.
   *
   * @param source the source
   * @return the markdown notebook output
   */
  public static MarkdownNotebookOutput get(final Object source) {
    try {
      final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
      final String className = null == source ? callingFrame.getClassName() : source.getClass().getCanonicalName();
      final String methodName = callingFrame.getMethodName();
      final String fileName = methodName + ".md";
      final File path = new File(Util.mkString(File.separator, "reports", className.replaceAll("\\.", "/").replaceAll("\\$", "/"), fileName));
      path.getParentFile().mkdirs();
      return new MarkdownNotebookOutput(path, methodName);
    } catch (final FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Add copy notebook output.
   *
   * @param out the out
   * @return the notebook output
   */
  @Override
  public NotebookOutput addCopy(final PrintStream out) {
    outs.add(out);
    return this;
  }
  
  /**
   * Anchor string.
   *
   * @return the string
   */
  public String anchor() {
    return String.format("<a name=\"p-%d\"></a>", anchor++);
  }
  
  @Override
  public void close() throws IOException {
    if (null != primaryOut) {
      primaryOut.close();
    }
  }
  
  @SuppressWarnings("unchecked")
  @Override
  public <T> T code(final UncheckedSupplier<T> fn, final int maxLog, final int framesNo) {
    try {
      final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[framesNo];
      final String sourceCode = CodeUtil.getInnerText(callingFrame);
      final SysOutInterceptor.LoggedResult<TimedResult<Object>> result = SysOutInterceptor.withOutput(() -> {
        try {
          return TimedResult.time(() -> fn.get());
        } catch (final Throwable e) {
          return new TimedResult<Object>(e, 0);
        }
      });
      out(anchor() + "Code from [%s:%s](%s#L%s) executed in %.2f seconds: ",
        callingFrame.getFileName(), callingFrame.getLineNumber(),
        linkTo(CodeUtil.findFile(callingFrame)), callingFrame.getLineNumber(), result.obj.seconds());
      out("```java");
      out("  " + sourceCode.replaceAll("\n", "\n  "));
      out("```");
      
      if (!result.log.isEmpty()) {
        out(anchor() + "Logging: ");
        out("```");
        out("    " + summarize(result.log, maxLog).replaceAll("\n", "\n    ").replaceAll("    ~", ""));
        out("```");
      }
      out("");
      
      final Object eval = result.obj.result;
      if (null != eval) {
        out(anchor() + "Returns: \n");
        String str;
        boolean escape;
        if (eval instanceof Throwable) {
          final ByteArrayOutputStream out = new ByteArrayOutputStream();
          ((Throwable) eval).printStackTrace(new PrintStream(out));
          str = new String(out.toByteArray(), "UTF-8");
          escape = true;//
        }
        else if (eval instanceof Component) {
          str = image(Util.toImage((Component) eval), "Result");
          escape = false;
        }
        else if (eval instanceof BufferedImage) {
          str = image((BufferedImage) eval, "Result");
          escape = false;
        }
        else if (eval instanceof TableOutput) {
          str = ((TableOutput) eval).toTextTable();
          escape = false;
        }
        else {
          str = eval.toString();
          escape = true;
        }
        if (escape) {
          out("```");
        }
        out(escape ? "    " + summarize(str, maxLog).replaceAll("\n", "\n    ").replaceAll("    ~", "") : str);
        if (escape) {
          out("```");
        }
        out("\n\n");
        if (eval instanceof Throwable) {
          throw new RuntimeException((Throwable) result.obj.result);
        }
      }
      return (T) eval;
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public OutputStream file(final String name) {
    try {
      return new FileOutputStream(new File(getResourceDir(), name));
    } catch (final FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public String file(final String data, final String caption) {
    return file(data, ++MarkdownNotebookOutput.excerptNumber + ".txt", caption);
  }
  
  @Override
  public String file(final String data, final String fileName, final String caption) {
    try {
      if (null != data) {
        IOUtils.write(data, new FileOutputStream(new File(getResourceDir(), fileName)), Charset.forName("UTF-8"));
      }
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
    return "[" + caption + "](etc/" + fileName + ")";
  }
  
  /**
   * Gets absolute url.
   *
   * @return the absolute url
   */
  public String getAbsoluteUrl() {
    return absoluteUrl;
  }
  
  /**
   * Sets absolute url.
   *
   * @param absoluteUrl the absolute url
   * @return the absolute url
   */
  public MarkdownNotebookOutput setAbsoluteUrl(final String absoluteUrl) {
    this.absoluteUrl = absoluteUrl;
    return this;
  }
  
  /**
   * Gets resource dir.
   *
   * @return the resource dir
   */
  public File getResourceDir() {
    final File etc = new File(fileName.getParentFile(), "etc");
    etc.mkdirs();
    return etc;
  }
  
  @Override
  public void h1(final String fmt, final Object... args) {
    out("# " + anchor() + fmt, args);
  }
  
  @Override
  public void h2(final String fmt, final Object... args) {
    out("## " + anchor() + fmt, args);
  }
  
  @Override
  public void h3(final String fmt, final Object... args) {
    out("### " + anchor() + fmt, args);
  }
  
  @Override
  public String image(final BufferedImage rawImage, final String caption) throws IOException {
    if (null == rawImage) return "";
    new ByteArrayOutputStream();
    final int thisImage = ++MarkdownNotebookOutput.imageNumber;
    final String fileName = name + "." + thisImage + ".png";
    final File file = new File(getResourceDir(), fileName);
    final BufferedImage stdImage = Util.resize(rawImage);
    if (stdImage != rawImage) {
      final String rawName = name + "_raw." + thisImage + ".png";
      ImageIO.write(rawImage, "png", new File(getResourceDir(), rawName));
    }
    ImageIO.write(stdImage, "png", file);
    return anchor() + "![" + caption + "](etc/" + file.getName() + ")";
  }
  
  @Override
  public String link(final File file, final String text) {
    try {
      return "[" + text + "](" + fileName.getCanonicalFile().toPath().relativize(file.getCanonicalFile().toPath()).normalize().toString().replaceAll("\\\\", "/") + ")";
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Link to string.
   *
   * @param file the file
   * @return the string
   * @throws IOException the io exception
   */
  public String linkTo(final File file) throws IOException {
    String path = Util.pathTo(fileName.getParentFile().getCanonicalFile(), file.getCanonicalFile());
    if (null != getAbsoluteUrl()) {
      path = new File(getAbsoluteUrl()).toPath().relativize(new File(path).toPath()).toString();
    }
    return path;
  }
  
  @Override
  public void out(final String fmt, final Object... args) {
    final String msg = 0 == args.length ? fmt : String.format(fmt, args);
    outs.forEach(out -> out.println(msg));
  }
  
  @Override
  public void p(final String fmt, final Object... args) {
    out(anchor() + fmt + "\n", args);
  }
  
  /**
   * Summarize string.
   *
   * @param logSrc the log src
   * @param maxLog the max log
   * @return the string
   */
  public String summarize(String logSrc, final int maxLog) {
    if (logSrc.length() > maxLog * 2) {
      final String prefix = logSrc.substring(0, maxLog);
      logSrc = prefix + String.format(
        (prefix.endsWith("\n") ? "" : "\n") + "~```\n~..." + file(logSrc, "skipping %s bytes") + "...\n~```\n",
        logSrc.length() - 2 * maxLog) + logSrc.substring(logSrc.length() - maxLog);
    }
    return logSrc;
  }
}
