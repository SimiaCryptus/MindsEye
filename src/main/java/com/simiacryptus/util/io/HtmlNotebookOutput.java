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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.Base64;
import java.util.UUID;

/**
 * The type Html notebook output.
 */
public class HtmlNotebookOutput implements NotebookOutput {
  private static final Logger log = LoggerFactory.getLogger(HtmlNotebookOutput.class);
  
  /**
   * The constant DEFAULT_ROOT.
   */
  public static String DEFAULT_ROOT = "https://github.com/SimiaCryptus/utilities/tree/master/";
  /**
   * The Working dir.
   */
  public final File workingDir;
  private final PrintStream primaryOut;
  /**
   * The Source root.
   */
  public String sourceRoot = HtmlNotebookOutput.DEFAULT_ROOT;
  /**
   * The Excerpt number.
   */
  int excerptNumber = 0;
  private int maxOutSize = 8 * 1024;
  
  /**
   * Instantiates a new Html notebook output.
   *
   * @param parentDirectory the parent directory
   * @param out             the out
   * @throws FileNotFoundException the file not found exception
   */
  public HtmlNotebookOutput(final File parentDirectory, final OutputStream out) throws FileNotFoundException {
    primaryOut = new PrintStream(out);
    workingDir = parentDirectory;
    out("<html><head><style>\n" +
          "pre {\n" +
          "    background-color: lightyellow;\n" +
          "    margin-left: 20pt;\n" +
          "    font-family: monospace;\n" +
          "}\n" +
          "</style></head><body>");
  }
  
  /**
   * Create html notebook output.
   *
   * @param parentDirectory the parent directory
   * @return the html notebook output
   * @throws FileNotFoundException the file not found exception
   */
  public static HtmlNotebookOutput create(final File parentDirectory) throws FileNotFoundException {
    final FileOutputStream out = new FileOutputStream(new File(parentDirectory, "index.html"));
    return new HtmlNotebookOutput(parentDirectory, out) {
      @Override
      public void close() throws IOException {
        out.close();
      }
    };
  }
  
  
  @Override
  public void close() throws IOException {
    out("</body></html>");
    if (null != primaryOut) {
      primaryOut.close();
    }
  }
  
  @SuppressWarnings("unchecked")
  @Override
  public <T> T code(final UncheckedSupplier<T> fn, final int maxLog, final int framesNo) {
    try {
      final StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
      final StackTraceElement callingFrame = stackTrace[framesNo];
      final String sourceCode = CodeUtil.getInnerText(callingFrame);
      final SysOutInterceptor.LoggedResult<TimedResult<Object>> result = SysOutInterceptor.withOutput(() -> {
        try {
          return TimedResult.time(() -> fn.get());
        } catch (final Throwable e) {
          return new TimedResult<Object>(e, 0);
        }
      });
      try {
        final URI resolved = URI.create(sourceRoot).resolve(Util.pathTo(CodeUtil.projectRoot, CodeUtil.findFile(callingFrame)));
        out("<p>Code from <a href='%s#L%s'>%s:%s</a> executed in %.2f seconds: <br/>",
            resolved, callingFrame.getLineNumber(), callingFrame.getFileName(), callingFrame.getLineNumber(), result.obj.seconds());
      } catch (final Exception e) {
        out("<p>Code from %s:%s executed in %.2f seconds: <br/>",
            callingFrame.getFileName(), callingFrame.getLineNumber(), result.obj.seconds());
      }
      out("<pre>");
      out(sourceCode);
      out("</pre>");
      
      if (!result.log.isEmpty()) {
        out("Logging: <br/>");
        out("<pre>");
        out(summarize(maxLog, result.log));
        out("</pre>");
      }
      out("");
      
      final Object eval = result.obj.result;
      if (null != eval) {
        out("Returns: <br/>");
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
          str = ((TableOutput) eval).toHtmlTable();
          escape = false;
        }
        else {
          str = eval.toString();
          escape = true;
        }
        if (escape) {
          out("<pre>" + summarize(maxLog, str) + "</pre>");
        }
        else {
          out(summarize(maxLog, str));
        }
        if (escape) {
        }
        out("\n\n");
        if (eval instanceof Throwable) {
          throw new RuntimeException((Throwable) result.obj.result);
        }
      }
      out("</p>");
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
    return file(data, excerptNumber++ + ".txt", caption);
  }
  
  @Override
  public String file(byte[] data, String filename, String caption) {
    try {
      IOUtils.write(data, new FileOutputStream(new File(getResourceDir(), filename)));
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
    return "<a href='etc/" + filename + "'>" + caption + "</a>";
  }
  
  @Override
  public String file(final String data, final String fileName, final String caption) {
    try {
      IOUtils.write(data, new FileOutputStream(new File(getResourceDir(), fileName)), Charset.forName("UTF-8"));
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
    return "<a href='etc/" + fileName + "'>" + caption + "</a>";
  }
  
  /**
   * Gets resource dir.
   *
   * @return the resource dir
   */
  public File getResourceDir() {
    final File etc = new File(workingDir, "etc");
    etc.mkdirs();
    return etc;
  }
  
  @Override
  public NotebookOutput setMaxOutSize(int size) {
    this.maxOutSize = size;
    return this;
  }
  
  /**
   * Gets source root.
   *
   * @return the source root
   */
  public String getSourceRoot() {
    return sourceRoot;
  }
  
  /**
   * Sets source root.
   *
   * @param sourceRoot the source root
   * @return the source root
   */
  public HtmlNotebookOutput setSourceRoot(final String sourceRoot) {
    this.sourceRoot = sourceRoot;
    return this;
  }
  
  @Override
  public void h1(final String fmt, final Object... args) {
    out("<h1>" + fmt + "</h1>", args);
  }
  
  @Override
  public void h2(final String fmt, final Object... args) {
    out("<h2>" + fmt + "</h2>", args);
  }
  
  @Override
  public void h3(final String fmt, final Object... args) {
    out("<h3>" + fmt + "</h3>", args);
  }
  
  @Override
  public String image(final BufferedImage rawImage, final String caption) throws IOException {
    if (null == rawImage) return "";
    new ByteArrayOutputStream();
    final String thisImage = UUID.randomUUID().toString().substring(0, 8);
    final File file = new File(getResourceDir(), "img" + thisImage + ".png");
    final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    ImageIO.write(rawImage, "png", buffer);
    final String pngSrc = Base64.getEncoder().encodeToString(buffer.toByteArray());
    if (pngSrc.length() < 4 * 1024) {
      return "<img src='data:image/png;base64," + pngSrc + "' alt='" + caption + "'/>";
    }
    else {
      final BufferedImage stdImage = Util.resize(rawImage);
      if (stdImage != rawImage) {
        ImageIO.write(rawImage, "png", new File(getResourceDir(), "raw" + thisImage + ".png"));
      }
      ImageIO.write(stdImage, "png", file);
      return "<img src='etc/" + file.getName() + "' alt='" + caption + "'/>";
    }
  }
  
  @Override
  public String link(final File file, final String text) {
    String path = null;
    try {
      path = workingDir.getCanonicalFile().toPath().relativize(file.getCanonicalFile().toPath()).normalize().toString().replaceAll("\\\\", "/");
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
    return String.format("<a href=\"%s\">%s</a>", path, text);
  }
  
  @Override
  public void out(final String fmt, final Object... args) {
    final String msg = 0 == args.length ? fmt : String.format(fmt, args);
    primaryOut.println(msg);
    primaryOut.flush();
    log.info(msg);
  }
  
  @Override
  public void p(final String fmt, final Object... args) {
    out("<p>" + fmt + "</p>", args);
  }
  
  @Override
  public String getFrontMatterProperty(String key) {
    return null;
  }
  
  @Override
  public String getName() {
    return "www";
  }
  
  /**
   * Summarize string.
   *
   * @param maxLog the max log
   * @param string the string
   * @return the string
   */
  public String summarize(final int maxLog, final String string) {
    if (string.length() > maxLog * 2) {
      final String left = string.substring(0, maxLog);
      final String right = string.substring(string.length() - maxLog);
      final String link = String.format(file(string, "\n...skipping %s bytes...\n"), string.length() - 2 * maxLog);
      return left + link + right;
    }
    else {
      return string;
    }
  }
  
  public int getMaxOutSize() {
    return maxOutSize;
  }
}
