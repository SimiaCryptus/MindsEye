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

package com.simiacryptus.util;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.util.io.BinaryChunkIterator;
import com.simiacryptus.util.io.TeeInputStream;
import com.simiacryptus.util.test.LabeledObject;
import org.apache.commons.io.output.ByteArrayOutputStream;

import javax.imageio.ImageIO;
import javax.net.ssl.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.X509Certificate;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

/**
 * The type Util.
 */
public class Util {
  
  /**
   * The constant R.
   */
  public static final ThreadLocal<Random> R = new ThreadLocal<Random>() {
    public final Random r = new Random(System.nanoTime());
    
    @Override
    protected Random initialValue() {
      return new Random(r.nextLong());
    }
  };
  private static final java.util.concurrent.atomic.AtomicInteger idcounter = new java.util.concurrent.atomic.AtomicInteger(0);
  private static final String jvmId = UUID.randomUUID().toString();
  
  /**
   * Add.
   *
   * @param f    the f
   * @param data the data
   */
  public static void add(final DoubleSupplier f, final double[] data) {
    for (int i = 0; i < data.length; i++) {
      data[i] += f.getAsDouble();
    }
  }
  
  /**
   * Binary stream stream.
   *
   * @param path       the path
   * @param name       the name
   * @param skip       the skip
   * @param recordSize the record size
   * @return the stream
   * @throws IOException the io exception
   */
  public static Stream<byte[]> binaryStream(final String path, final String name, final int skip, final int recordSize) throws IOException {
    final File file = new File(path, name);
    final byte[] fileData = org.apache.commons.io.IOUtils.toByteArray(new BufferedInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(file)))));
    final DataInputStream in = new DataInputStream(new ByteArrayInputStream(fileData));
    in.skip(skip);
    return Util.toIterator(new BinaryChunkIterator(in, recordSize));
  }
  
  /**
   * Cache function.
   *
   * @param <F>   the type parameter
   * @param <T>   the type parameter
   * @param inner the inner
   * @return the function
   */
  public static <F, T> Function<F, T> cache(final Function<F, T> inner) {
    final LoadingCache<F, T> cache = CacheBuilder.newBuilder().build(new CacheLoader<F, T>() {
      @Override
      public T load(final F key) throws Exception {
        return inner.apply(key);
      }
    });
    return cache::apply;
  }
  
  /**
   * Cache input stream.
   *
   * @param url  the url
   * @param file the file
   * @return the input stream
   * @throws IOException              the io exception
   * @throws NoSuchAlgorithmException the no such algorithm exception
   * @throws KeyStoreException        the key store exception
   * @throws KeyManagementException   the key management exception
   */
  public static InputStream cache(final String url, final String file) throws IOException, NoSuchAlgorithmException, KeyStoreException, KeyManagementException {
    if (new File(file).exists()) {
      return new FileInputStream(file);
    }
    else {
      final TrustManager[] trustManagers = {
        new X509TrustManager() {
          @Override
          public void checkClientTrusted(
            final X509Certificate[] certs, final String authType) {
          }
          
          @Override
          public void checkServerTrusted(
            final X509Certificate[] certs, final String authType) {
          }
          
          @Override
          public X509Certificate[] getAcceptedIssuers() {
            return new X509Certificate[0];
          }
        }
      };
      final SSLContext ctx = SSLContext.getInstance("TLS");
      ctx.init(null, trustManagers, null);
      final SSLSocketFactory sslFactory = ctx.getSocketFactory();
      final URLConnection urlConnection = new URL(url).openConnection();
      if (urlConnection instanceof HttpsURLConnection) {
        final HttpsURLConnection conn = (HttpsURLConnection) urlConnection;
        conn.setSSLSocketFactory(sslFactory);
        conn.setRequestMethod("GET");
      }
      final InputStream inputStream = urlConnection.getInputStream();
      final FileOutputStream cache = new FileOutputStream(file);
      return new TeeInputStream(inputStream, cache);
    }
  }
  
  /**
   * Cache input stream.
   *
   * @param url the url
   * @return the input stream
   * @throws IOException              the io exception
   * @throws NoSuchAlgorithmException the no such algorithm exception
   * @throws KeyStoreException        the key store exception
   * @throws KeyManagementException   the key management exception
   */
  public static InputStream cache(final URI url) throws IOException, NoSuchAlgorithmException, KeyStoreException, KeyManagementException {
    return Util.cache(url.toString(), new File(url.getPath()).getName());
  }
  
  /**
   * Current stack string [ ].
   *
   * @return the string [ ]
   */
  public static String[] currentStack() {
    return Stream.of(Thread.currentThread().getStackTrace()).map(Object::toString).toArray(i -> new String[i]);
  }
  
  /**
   * Cvt temporal unit.
   *
   * @param units the units
   * @return the temporal unit
   */
  public static TemporalUnit cvt(final TimeUnit units) {
    switch (units) {
      case DAYS:
        return ChronoUnit.DAYS;
      case HOURS:
        return ChronoUnit.HOURS;
      case MINUTES:
        return ChronoUnit.MINUTES;
      case SECONDS:
        return ChronoUnit.SECONDS;
      case NANOSECONDS:
        return ChronoUnit.NANOS;
      case MICROSECONDS:
        return ChronoUnit.MICROS;
      case MILLISECONDS:
        return ChronoUnit.MILLIS;
      default:
        throw new IllegalArgumentException(units.toString());
    }
  }
  
  /**
   * Gets last.
   *
   * @param <T>    the type parameter
   * @param stream the stream
   * @return the last
   */
  public static <T> T getLast(final Stream<T> stream) {
    final List<T> collect = stream.collect(Collectors.toList());
    final T last = collect.get(collect.size() - 1);
    return last;
  }
  
  /**
   * Layout.
   *
   * @param c the c
   */
  public static void layout(final Component c) {
    c.doLayout();
    if (c instanceof Container) {
      Arrays.stream(((Container) c).getComponents()).forEach(Util::layout);
    }
  }
  
  /**
   * Mk string string.
   *
   * @param separator the separator
   * @param strs      the strs
   * @return the string
   */
  public static String mkString(final String separator, final String... strs) {
    return Arrays.asList(strs).stream().collect(Collectors.joining(separator));
  }
  
  /**
   * Path to string.
   *
   * @param from the from
   * @param to   the to
   * @return the string
   */
  public static String pathTo(final File from, final File to) {
    return from.toPath().relativize(to.toPath()).toString().replaceAll("\\\\", "/");
  }
  
  /**
   * Read byte [ ].
   *
   * @param i the
   * @param s the s
   * @return the byte [ ]
   * @throws IOException the io exception
   */
  public static byte[] read(final DataInputStream i, final int s) throws IOException {
    final byte[] b = new byte[s];
    int pos = 0;
    while (b.length > pos) {
      final int read = i.read(b, pos, b.length - pos);
      if (0 == read) {
        throw new RuntimeException();
      }
      pos += read;
    }
    return b;
  }
  
  /**
   * Report.
   *
   * @param fragments the fragments
   * @throws IOException the io exception
   */
  public static void report(final Stream<String> fragments) throws IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Util.getLast(Arrays.stream(Thread.currentThread().getStackTrace())//
      .filter(x -> x.getClassName().contains("simiacryptus")));
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    fragments.forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }
  
  /**
   * Report.
   *
   * @param fragments the fragments
   * @throws IOException the io exception
   */
  public static void report(final String... fragments) throws IOException {
    Util.report(Stream.of(fragments));
  }
  
  /**
   * Resize buffered image.
   *
   * @param image the image
   * @return the buffered image
   */
  public static BufferedImage resize(final BufferedImage image) {
    if (null == image) return image;
    final int width = Math.min(image.getWidth(), 800);
    if (width == image.getWidth()) return image;
    final int height = image.getHeight() * width / image.getWidth();
    final BufferedImage rerender = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    final Graphics gfx = rerender.getGraphics();
    final RenderingHints hints = new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
    ((Graphics2D) gfx).setRenderingHints(hints);
    gfx.drawImage(image, 0, 0, rerender.getWidth(), rerender.getHeight(), null);
    return rerender;
  }
  
  /**
   * To image buffered image.
   *
   * @param component the component
   * @return the buffered image
   */
  public static BufferedImage toImage(final Component component) {
    try {
      Util.layout(component);
      final BufferedImage img = new BufferedImage(component.getWidth(), component.getHeight(), BufferedImage.TYPE_INT_ARGB_PRE);
      final Graphics2D g = img.createGraphics();
      g.setColor(component.getForeground());
      g.setFont(component.getFont());
      component.print(g);
      return img;
    } catch (final Exception e) {
      return null;
    }
  }
  
  /**
   * To inline image string.
   *
   * @param img the img
   * @param alt the alt
   * @return the string
   */
  public static String toInlineImage(final BufferedImage img, final String alt) {
    return Util.toInlineImage(new LabeledObject<>(img, alt));
  }
  
  /**
   * To inline image string.
   *
   * @param img the img
   * @return the string
   */
  public static String toInlineImage(final LabeledObject<BufferedImage> img) {
    final ByteArrayOutputStream b = new ByteArrayOutputStream();
    try {
      ImageIO.write(img.data, "PNG", b);
    } catch (final Exception e) {
      throw new RuntimeException(e);
    }
    final byte[] byteArray = b.toByteArray();
    final String encode = Base64.getEncoder().encodeToString(byteArray);
    return "<img src=\"data:image/png;base64," + encode + "\" alt=\"" + img.label + "\" />";
  }
  
  /**
   * To iterator stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @return the stream
   */
  public static <T> Stream<T> toIterator(final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }
  
  /**
   * To stream stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @return the stream
   */
  public static <T> Stream<T> toStream(final Iterator<T> iterator) {
    return Util.toStream(iterator, 0);
  }
  
  /**
   * To stream stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @param size     the size
   * @return the stream
   */
  public static <T> Stream<T> toStream(final Iterator<T> iterator, final int size) {
    return Util.toStream(iterator, size, false);
  }
  
  /**
   * To stream stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @param size     the size
   * @param parallel the parallel
   * @return the stream
   */
  public static <T> Stream<T> toStream(final Iterator<T> iterator, final int size, final boolean parallel) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), parallel);
  }
  
  /**
   * Uuid uuid.
   *
   * @return the uuid
   */
  public static UUID uuid() {
    String index = Integer.toHexString(Util.idcounter.incrementAndGet());
    while (index.length() < 8) {
      index = "0" + index;
    }
    final String tempId = Util.jvmId.substring(0, Util.jvmId.length() - index.length()) + index;
    return UUID.fromString(tempId);
  }
}
