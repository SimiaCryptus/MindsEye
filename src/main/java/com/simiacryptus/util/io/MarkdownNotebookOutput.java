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
  
  private static int imageNumber = 0;
  private static int excerptNumber = 0;
  
  private final List<PrintStream> outs = new ArrayList<>();
  private final File fileName;
  private final String name;
  private final OutputStream primaryOut;
  
  /**
   * Instantiates a new Markdown notebook output.
   *
   * @param fileName the file name
   * @param name     the name
   * @throws FileNotFoundException the file not found exception
   */
  public MarkdownNotebookOutput(File fileName, String name) throws FileNotFoundException {
    this.name = name;
    this.primaryOut = new FileOutputStream(fileName);
    outs.add(new PrintStream(primaryOut));
    this.fileName = fileName;
  }
  
  /**
   * Get markdown notebook output.
   *
   * @param source the source
   * @return the markdown notebook output
   */
  public static MarkdownNotebookOutput get(Object source) {
    try {
      StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
      String className = null == source ? callingFrame.getClassName() : source.getClass().getCanonicalName();
      String methodName = callingFrame.getMethodName();
      String fileName = methodName + ".md";
      File path = new File(Util.mkString(File.separator, "reports", className.replaceAll("\\.", "/").replaceAll("\\$", "/"), fileName));
      path.getParentFile().mkdirs();
      return new MarkdownNotebookOutput(path, methodName);
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public OutputStream file(String name) {
    try {
      return new FileOutputStream(new File(getResourceDir(), name));
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Add copy notebook output.
   *
   * @param out the out
   * @return the notebook output
   */
  public NotebookOutput addCopy(PrintStream out) {
    outs.add(out);
    return this;
  }
  
  @Override
  public void out(String fmt, Object... args) {
    String msg = 0 == args.length ? fmt : String.format(fmt, args);
    outs.forEach(out -> out.println(msg));
  }
  
  @Override
  public void p(String fmt, Object... args) {
    this.out(fmt + "\n", args);
  }
  
  @Override
  public void h1(String fmt, Object... args) {
    this.out("# " + fmt, args);
  }
  
  @Override
  public void h2(String fmt, Object... args) {
    this.out("## " + fmt, args);
  }
  
  @Override
  public void h3(String fmt, Object... args) {
    this.out("### " + fmt, args);
  }
  
  @Override
  public <T> T code(UncheckedSupplier<T> fn, int maxLog, int framesNo) {
    try {
      StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[framesNo];
      String sourceCode = CodeUtil.getInnerText(callingFrame);
      SysOutInterceptor.LoggedResult<TimedResult<Object>> result = SysOutInterceptor.withOutput(() -> {
        try {
          return TimedResult.time(() -> fn.get());
        } catch (Throwable e) {
          return new TimedResult(e, 0);
        }
      });
      File callingFile = CodeUtil.findFile(callingFrame).getCanonicalFile();
      File contextPath = fileName.getParentFile().getCanonicalFile();
      String relativePath = Util.pathTo(contextPath, callingFile);
      out("Code from [%s:%s](%s#L%s) executed in %.2f seconds: ",
        callingFrame.getFileName(), callingFrame.getLineNumber(),
        relativePath, callingFrame.getLineNumber(), result.obj.seconds());
      out("```java");
      out("  " + sourceCode.replaceAll("\n", "\n  "));
      out("```");
      
      if (!result.log.isEmpty()) {
        out("Logging: ");
        out("```");
        out("    " + summarize(result.log, maxLog).replaceAll("\n", "\n    ").replaceAll("    ~", ""));
        out("```");
      }
      out("");
      
      Object eval = result.obj.result;
      if (null != eval) {
        out("Returns: \n");
        String str;
        boolean escape;
        if (eval instanceof Throwable) {
          ByteArrayOutputStream out = new ByteArrayOutputStream();
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
        if (escape) out("```");
        out(escape ? ("    " + summarize(str, maxLog).replaceAll("\n", "\n    ").replaceAll("    ~", "")) : str);
        if (escape) out("```");
        out("\n\n");
        if (eval instanceof Throwable) {
          throw new RuntimeException((Throwable) result.obj.result);
        }
      }
      return (T) eval;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Summarize string.
   *
   * @param logSrc the log src
   * @param maxLog the max log
   * @return the string
   */
  public String summarize(String logSrc, int maxLog) {
    if (logSrc.length() > maxLog * 2) {
      String prefix = logSrc.substring(0, maxLog);
      logSrc = prefix + String.format(
        (prefix.endsWith("\n") ? "" : "\n") + "~```\n~..." + file(logSrc, "skipping %s bytes") + "...\n~```\n",
        logSrc.length() - 2 * maxLog) + logSrc.substring(logSrc.length() - maxLog);
    }
    else if (logSrc.length() > 0) {
      logSrc = logSrc;
    }
    return logSrc;
  }
  
  @Override
  public String file(String data, String caption) {
    return file(data, ++excerptNumber + ".txt", caption);
  }
  
  @Override
  public String file(String data, String fileName, String caption) {
    try {
      if (null != data) {
        IOUtils.write(data, new FileOutputStream(new File(getResourceDir(), fileName)), Charset.forName("UTF-8"));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return "[" + caption + "](etc/" + fileName + ")";
  }
  
  @Override
  public String image(BufferedImage rawImage, String caption) throws IOException {
    if (null == rawImage) return "";
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    int thisImage = ++imageNumber;
    String fileName = this.name + "." + thisImage + ".png";
    File file = new File(getResourceDir(), fileName);
    BufferedImage stdImage = Util.resize(rawImage);
    if (stdImage != rawImage) {
      String rawName = this.name + "_raw." + thisImage + ".png";
      ImageIO.write(rawImage, "png", new File(getResourceDir(), rawName));
    }
    ImageIO.write(stdImage, "png", file);
    return "![" + caption + "](etc/" + file.getName() + ")";
  }
  
  /**
   * Gets resource dir.
   *
   * @return the resource dir
   */
  public File getResourceDir() {
    File etc = new File(this.fileName.getParentFile(), "etc");
    etc.mkdirs();
    return etc;
  }
  
  @Override
  public void close() throws IOException {
    if (null != primaryOut) primaryOut.close();
  }
  
}
