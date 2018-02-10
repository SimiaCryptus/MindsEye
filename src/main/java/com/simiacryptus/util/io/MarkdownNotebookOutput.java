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

package com.simiacryptus.util.io;

import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.lang.CodeUtil;
import com.simiacryptus.util.lang.TimedResult;
import com.simiacryptus.util.lang.UncheckedSupplier;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringEscapeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.management.ManagementFactory;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The type Markdown notebook output.
 */
public class MarkdownNotebookOutput implements NotebookOutput {
  /**
   * The Logger.
   */
  static final Logger log = LoggerFactory.getLogger(MarkdownNotebookOutput.class);
  
  private static int excerptNumber = 0;
  private static int imageNumber = 0;
  @javax.annotation.Nonnull
  private final File fileName;
  private final String name;
  @javax.annotation.Nonnull
  private final PrintStream primaryOut;
  private final List<String> buffer = new ArrayList<>();
  private final Map<String, String> frontMatter = new HashMap<>();
  /**
   * The Toc.
   */
  @javax.annotation.Nonnull
  public List<String> toc = new ArrayList<>();
  /**
   * The Anchor.
   */
  int anchor = 0;
  @Nullable
  private String absoluteUrl = null;
  private int maxOutSize = 8 * 1024;
  
  /**
   * Instantiates a new Markdown notebook output.
   *
   * @param fileName the file name
   * @param name     the name
   * @throws FileNotFoundException the file not found exception
   */
  public MarkdownNotebookOutput(@javax.annotation.Nonnull final File fileName, final String name) throws FileNotFoundException {
    this.name = name;
    primaryOut = new PrintStream(new FileOutputStream(fileName));
    this.fileName = fileName;
  }
  
  /**
   * Get markdown notebook output.
   *
   * @param sourceClass the source class
   * @param absoluteUrl the absolute url
   * @param suffix      the suffix
   * @return the markdown notebook output
   */
  @javax.annotation.Nonnull
  public static MarkdownNotebookOutput get(@javax.annotation.Nonnull Class<?> sourceClass, @Nullable String absoluteUrl, @javax.annotation.Nonnull String... suffix) {
    try {
      final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
      final String methodName = callingFrame.getMethodName();
      final String className = sourceClass.getCanonicalName();
      @javax.annotation.Nonnull File path = new File(Util.mkString(File.separator, "reports", className.replaceAll("\\.", "/").replaceAll("\\$", "/")));
      for (int i = 0; i < suffix.length - 1; i++) path = new File(path, suffix[i]);
      String testName = suffix.length == 0 ? methodName : suffix[suffix.length - 1];
      path = new File(path, testName + ".md");
      path.getParentFile().mkdirs();
      @javax.annotation.Nonnull MarkdownNotebookOutput notebookOutput = new MarkdownNotebookOutput(path, testName);
      if (null != absoluteUrl) {
        try {
          String url = new URI(absoluteUrl + "/" + path.toPath().toString().replaceAll("\\\\", "/")).normalize().toString();
          notebookOutput.setAbsoluteUrl(url);
        } catch (URISyntaxException e) {
          throw new RuntimeException(e);
        }
      }
      return notebookOutput;
    } catch (@javax.annotation.Nonnull final FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Get markdown notebook output.
   *
   * @return the markdown notebook output
   */
  public static MarkdownNotebookOutput get() {
    try {
      final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[2];
      final String className = callingFrame.getClassName();
      final String methodName = callingFrame.getMethodName();
      @javax.annotation.Nonnull final String fileName = methodName + ".md";
      @javax.annotation.Nonnull File path = new File(Util.mkString(File.separator, "reports", className.replaceAll("\\.", "/").replaceAll("\\$", "/")));
      path = new File(path, fileName);
      path.getParentFile().mkdirs();
      return new MarkdownNotebookOutput(path, methodName);
    } catch (@javax.annotation.Nonnull final FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public void close() throws IOException {
    if (null != primaryOut) {
      primaryOut.close();
      try (@javax.annotation.Nonnull PrintWriter out = new PrintWriter(new FileOutputStream(fileName))) {
        if (!frontMatter.isEmpty()) {
          out.println("---");
  
          frontMatter.forEach((key, value) -> {
            String escaped = StringEscapeUtils.escapeJson(value)
              .replaceAll("\n", " ")
              .replaceAll(":", "&#58;")
              .replaceAll("\\{", "\\{")
              .replaceAll("\\}", "\\}");
            out.println(String.format("%s: %s", key, escaped));
          });
          out.println("---");
        }
        toc.forEach(out::println);
        out.print("\n\n");
        buffer.forEach(out::println);
      }
    }
  }
  
  public void setFrontMatterProperty(String key, String value) {
    frontMatter.put(key, value);
  }
  
  @Override
  public String getFrontMatterProperty(String key) {
    return frontMatter.get(key);
  }
  
  @Override
  public String getName() {
    return name;
  }
  
  /**
   * Anchor string.
   *
   * @param anchorId the anchor id
   * @return the string
   */
  public String anchor(String anchorId) {
    return String.format("<a id=\"%s\"></a>", anchorId);
  }
  
  /**
   * Anchor id string.
   *
   * @return the string
   */
  public String anchorId() {
    return String.format("p-%d", anchor++);
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public <T> T code(@javax.annotation.Nonnull final UncheckedSupplier<T> fn, final int maxLog, final int framesNo) {
    try {
      final StackTraceElement callingFrame = Thread.currentThread().getStackTrace()[framesNo];
      final String sourceCode = CodeUtil.getInnerText(callingFrame);
      @javax.annotation.Nonnull final SysOutInterceptor.LoggedResult<TimedResult<Object>> result = SysOutInterceptor.withOutput(() -> {
        long priorGcMs = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum();
        final long start = System.nanoTime();
        try {
          @Nullable Object result1 = null;
          try {
            result1 = fn.get();
          } catch (@javax.annotation.Nonnull final RuntimeException e) {
            throw e;
          } catch (@javax.annotation.Nonnull final Exception e) {
            throw new RuntimeException(e);
          }
          long gcTime = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum() - priorGcMs;
          return new TimedResult<Object>(result1, System.nanoTime() - start, gcTime);
        } catch (@javax.annotation.Nonnull final Throwable e) {
          long gcTime = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum() - priorGcMs;
          return new TimedResult<Object>(e, System.nanoTime() - start, gcTime);
        }
      });
      out(anchor(anchorId()) + "Code from [%s:%s](%s#L%s) executed in %.2f seconds (%.3f gc): ",
        callingFrame.getFileName(), callingFrame.getLineNumber(),
        linkTo(CodeUtil.findFile(callingFrame)), callingFrame.getLineNumber(), result.obj.seconds(), result.obj.gc_seconds());
      String text = sourceCode.replaceAll("\n", "\n  ");
      out("```java");
      out("  " + text);
      out("```");
      
      if (!result.log.isEmpty()) {
        String summary = summarize(result.log, maxLog).replaceAll("\n", "\n    ").replaceAll("    ~", "");
        out(anchor(anchorId()) + "Logging: ");
        out("```");
        out("    " + summary);
        out("```");
      }
      out("");
      
      final Object eval = result.obj.result;
      if (null != eval) {
        out(anchor(anchorId()) + "Returns: \n");
        String str;
        boolean escape;
        if (eval instanceof Throwable) {
          @javax.annotation.Nonnull final ByteArrayOutputStream out = new ByteArrayOutputStream();
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
        @javax.annotation.Nonnull String fmt = escape ? "    " + summarize(str, maxLog).replaceAll("\n", "\n    ").replaceAll("    ~", "") : str;
        if (escape) {
          out("```");
          out(fmt);
          out("```");
        }
        else {
          out(fmt);
        }
        out("\n\n");
        if (eval instanceof RuntimeException) {
          throw ((RuntimeException) result.obj.result);
        }
        if (eval instanceof Throwable) {
          throw new RuntimeException((Throwable) result.obj.result);
        }
      }
      return (T) eval;
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @javax.annotation.Nonnull
  @Override
  public OutputStream file(@javax.annotation.Nonnull final String name) {
    try {
      return new FileOutputStream(new File(getResourceDir(), name));
    } catch (@javax.annotation.Nonnull final FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  @javax.annotation.Nonnull
  @Override
  public String file(final String data, final String caption) {
    return file(data, ++MarkdownNotebookOutput.excerptNumber + ".txt", caption);
  }
  
  @javax.annotation.Nonnull
  @Override
  public String file(@javax.annotation.Nonnull byte[] data, @javax.annotation.Nonnull String filename, String caption) {
    return file(new String(data, Charset.forName("UTF-8")), filename, caption);
  }
  
  @javax.annotation.Nonnull
  @Override
  public String file(@Nullable final String data, @javax.annotation.Nonnull final String fileName, final String caption) {
    try {
      if (null != data) {
        IOUtils.write(data, new FileOutputStream(new File(getResourceDir(), fileName)), Charset.forName("UTF-8"));
      }
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
    return "[" + caption + "](etc/" + fileName + ")";
  }
  
  /**
   * Gets absolute url.
   *
   * @return the absolute url
   */
  @Nullable
  public String getAbsoluteUrl() {
    return absoluteUrl;
  }
  
  /**
   * Sets absolute url.
   *
   * @param absoluteUrl the absolute url
   * @return the absolute url
   */
  @javax.annotation.Nonnull
  public MarkdownNotebookOutput setAbsoluteUrl(final String absoluteUrl) {
    this.absoluteUrl = absoluteUrl;
    return this;
  }
  
  /**
   * Gets resource dir.
   *
   * @return the resource dir
   */
  @javax.annotation.Nonnull
  public File getResourceDir() {
    @javax.annotation.Nonnull final File etc = new File(fileName.getParentFile(), "etc");
    etc.mkdirs();
    return etc;
  }
  
  @javax.annotation.Nonnull
  @Override
  public NotebookOutput setMaxOutSize(int size) {
    this.maxOutSize = size;
    return this;
  }
  
  @Override
  public void h1(@javax.annotation.Nonnull final String fmt, final Object... args) {
    String anchorId = anchorId();
    @javax.annotation.Nonnull String msg = format(fmt, args);
    toc.add(String.format("1. [%s](#%s)", msg, anchorId));
    out("# " + anchor(anchorId) + msg);
  }
  
  @Override
  public void h2(@javax.annotation.Nonnull final String fmt, final Object... args) {
    String anchorId = anchorId();
    @javax.annotation.Nonnull String msg = format(fmt, args);
    toc.add(String.format("   1. [%s](#%s)", msg, anchorId));
    out("## " + anchor(anchorId) + fmt, args);
  }
  
  @Override
  public void h3(@javax.annotation.Nonnull final String fmt, final Object... args) {
    String anchorId = anchorId();
    @javax.annotation.Nonnull String msg = format(fmt, args);
    toc.add(String.format("      1. [%s](#%s)", msg, anchorId));
    out("### " + anchor(anchorId) + fmt, args);
  }
  
  @javax.annotation.Nonnull
  @Override
  public String image(@Nullable final BufferedImage rawImage, final String caption) throws IOException {
    if (null == rawImage) return "";
    new ByteArrayOutputStream();
    final int thisImage = ++MarkdownNotebookOutput.imageNumber;
    @javax.annotation.Nonnull final String fileName = name + "." + thisImage + ".png";
    @javax.annotation.Nonnull final File file = new File(getResourceDir(), fileName);
    @javax.annotation.Nullable final BufferedImage stdImage = Util.resize(rawImage);
    if (stdImage != rawImage) {
      @javax.annotation.Nonnull final String rawName = name + "_raw." + thisImage + ".png";
      ImageIO.write(rawImage, "png", new File(getResourceDir(), rawName));
    }
    ImageIO.write(stdImage, "png", file);
    return anchor(anchorId()) + "![" + caption + "](etc/" + file.getName() + ")";
  }
  
  @javax.annotation.Nonnull
  @Override
  public String link(@javax.annotation.Nonnull final File file, final String text) {
    try {
      return "[" + text + "](" + codeFile(file) + ")";
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Code file string.
   *
   * @param file the file
   * @return the string
   * @throws IOException the io exception
   */
  public String codeFile(@javax.annotation.Nonnull File file) throws IOException {
    Path path = pathToCodeFile(file);
    if (null != getAbsoluteUrl()) {
      try {
        return new URI(getAbsoluteUrl()).resolve(path.normalize().toString().replaceAll("\\\\", "/")).normalize().toString();
      } catch (URISyntaxException e) {
        throw new RuntimeException(e);
      }
    }
    else {
      return path.normalize().toString().replaceAll("\\\\", "/");
    }
  }
  
  /**
   * Path to code file path.
   *
   * @param file the file
   * @return the path
   * @throws IOException the io exception
   */
  public Path pathToCodeFile(@javax.annotation.Nonnull File file) throws IOException {
    return fileName.getCanonicalFile().toPath().relativize(file.getCanonicalFile().toPath());
  }
  
  /**
   * Link to string.
   *
   * @param file the file
   * @return the string
   * @throws IOException the io exception
   */
  public String linkTo(@javax.annotation.Nonnull final File file) throws IOException {
    return codeFile(file);
  }
  
  @Override
  public void out(@javax.annotation.Nonnull final String fmt, final Object... args) {
    @javax.annotation.Nonnull final String msg = format(fmt, args);
    buffer.add(msg);
    primaryOut.println(msg);
    log.info(msg);
  }
  
  /**
   * Format string.
   *
   * @param fmt  the fmt
   * @param args the args
   * @return the string
   */
  @javax.annotation.Nonnull
  public String format(@javax.annotation.Nonnull String fmt, @javax.annotation.Nonnull Object... args) {
    return 0 == args.length ? fmt : String.format(fmt, args);
  }
  
  @Override
  public void p(final String fmt, final Object... args) {
    out(anchor(anchorId()) + fmt + "\n", args);
  }
  
  /**
   * Summarize string.
   *
   * @param logSrc the log src
   * @param maxLog the max log
   * @return the string
   */
  @javax.annotation.Nonnull
  public String summarize(@javax.annotation.Nonnull String logSrc, final int maxLog) {
    if (logSrc.length() > maxLog * 2) {
      @javax.annotation.Nonnull final String prefix = logSrc.substring(0, maxLog);
      logSrc = prefix + String.format(
        (prefix.endsWith("\n") ? "" : "\n") + "~```\n~..." + file(logSrc, "skipping %s bytes") + "...\n~```\n",
        logSrc.length() - 2 * maxLog) + logSrc.substring(logSrc.length() - maxLog);
    }
    return logSrc;
  }
  
  public int getMaxOutSize() {
    return maxOutSize;
  }
  
}
