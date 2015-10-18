package com.simiacryptus.mindseye;

import java.awt.Desktop;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Base64;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

import javax.imageio.ImageIO;

import org.apache.commons.io.output.ByteArrayOutputStream;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.simiacryptus.mindseye.core.BinaryChunkIterator;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import de.javakaffee.kryoserializers.EnumMapSerializer;
import de.javakaffee.kryoserializers.EnumSetSerializer;
import de.javakaffee.kryoserializers.KryoReflectionFactorySupport;

public class Util {

  private final static java.util.concurrent.atomic.AtomicInteger idcounter = new java.util.concurrent.atomic.AtomicInteger(0);

  private final static String jvmId = UUID.randomUUID().toString();

  public static final ThreadLocal<Random> R = new ThreadLocal<Random>() {
    public final Random r = new Random(System.nanoTime());

    @Override
    protected Random initialValue() {
      return new Random(this.r.nextLong());
    }
  };

  private static final ThreadLocal<Kryo> threadKryo = new ThreadLocal<Kryo>() {

    @Override
    protected Kryo initialValue() {
      final Kryo kryo = new KryoReflectionFactorySupport() {

        @Override
        public Serializer<?> getDefaultSerializer(@SuppressWarnings("rawtypes") final Class clazz) {
          if (EnumSet.class.isAssignableFrom(clazz))
            return new EnumSetSerializer();
          if (EnumMap.class.isAssignableFrom(clazz))
            return new EnumMapSerializer();
          return super.getDefaultSerializer(clazz);
        }

      };
      return kryo;
    }

  };

  public static void add(final DoubleSupplier f, final double[] data) {
    for (int i = 0; i < data.length; i++) {
      data[i] += f.getAsDouble();
    }
  }

  public static Stream<byte[]> binaryStream(final String path, final String name, final int skip, final int recordSize) throws IOException {
    File file = new File(path, name);
    byte[] fileData = org.apache.commons.io.IOUtils.toByteArray(new java.io.BufferedInputStream(new GZIPInputStream(new java.io.BufferedInputStream(new FileInputStream(file)))));
    final DataInputStream in = new DataInputStream(new java.io.ByteArrayInputStream(fileData));
    in.skip(skip);
    return Util.toIterator(new BinaryChunkIterator(in, recordSize));
  }

  public static double bounds(final double value) {
    final int max = 0xFF;
    final int min = 0;
    return (value < min) ? min : ((value > max) ? max : value);
  }

  public static int bounds(final int value) {
    final int max = 0xFF;
    final int min = 0;
    return (value < min) ? min : ((value > max) ? max : value);
  }

  public static String[] currentStack() {
    return java.util.stream.Stream.of(Thread.currentThread().getStackTrace()).map(Object::toString).toArray(i -> new String[i]);
  }

  public static Kryo kryo() {
    return Util.threadKryo.get();
  }

  public static byte[] read(final DataInputStream i, final int s) throws IOException {
    final byte[] b = new byte[s];
    int pos = 0;
    while (b.length > pos) {
      final int read = i.read(b, pos, b.length - pos);
      if (0 == read)
        throw new RuntimeException();
      pos += read;
    }
    return b;
  }

  public static void report(final Stream<String> fragments) throws FileNotFoundException, IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = getLast(Arrays.stream(Thread.currentThread().getStackTrace())//
        .filter(x->x.getClassName().contains("simiacryptus")));
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    fragments.forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  public static <T> T getLast(Stream<T> stream) {
    List<T> collect = stream.collect(Collectors.toList());
    T last = collect.get(collect.size()-1);
    return last;
  }

  public static void report(final String... fragments) throws FileNotFoundException, IOException {
    Util.report(Stream.of(fragments));
  }

  public static boolean thermalStep(final double prev, final double next, final double temp) {
    if (next < prev)
      return true;
    if (temp <= 0.)
      return false;
    final double p = Math.exp(-(next - prev) / (Math.min(next, prev) * temp));
    final boolean step = Math.random() < p;
    return step;
  }

  public static NDArray fillImage(final byte[] b, final NDArray ndArray) {
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        ndArray.set(new int[] { x, y }, b[x + y * 28]&0xFF);
      }
    }
    return ndArray;
  }

  public static BufferedImage toImage(final NDArray ndArray) {
    final int[] dims = ndArray.getDims();
    final BufferedImage img = new BufferedImage(dims[0], dims[1], BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        if (ndArray.getDims()[2] == 1) {
          final double value = ndArray.get(x, y, 0);
          //int asInt = ((byte) value) & 0xFF;
          img.setRGB(x, y, bounds((int)value) * 0x010101);
        } else {
          final double red = Util.bounds(ndArray.get(x, y, 0));
          final double green = Util.bounds(ndArray.get(x, y, 1));
          final double blue = Util.bounds(ndArray.get(x, y, 2));
          img.setRGB(x, y, (int) (red + ((int) green << 8) + ((int) blue << 16)));
        }
      }
    }
    return img;
  }

  public static String toInlineImage(final BufferedImage img, final String alt) {
    return Util.toInlineImage(new LabeledObject<BufferedImage>(img, alt));
  }

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

  public static <T> Stream<T> toIterator(final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }

  public static <T> Stream<T> toStream(final Iterator<T> iterator) {
    return Util.toStream(iterator, 0);
  }

  public static <T> Stream<T> toStream(final Iterator<T> iterator, final int size) {
    return Util.toStream(iterator, size, false);
  }

  public static <T> Stream<T> toStream(final Iterator<T> iterator, final int size, final boolean parallel) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), parallel);
  }

  public static UUID uuid() {
    String index = Integer.toHexString(idcounter.incrementAndGet());
    while (index.length() < 8) {
      index = "0" + index;
    }
    final String tempId = jvmId.substring(0, jvmId.length() - index.length()) + index;
    return UUID.fromString(tempId);
  }
}
