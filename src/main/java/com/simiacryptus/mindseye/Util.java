package com.simiacryptus.mindseye;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.DoubleSupplier;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

import javax.imageio.ImageIO;

import org.apache.commons.io.output.ByteArrayOutputStream;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.simiacryptus.mindseye.data.BinaryChunkIterator;
import com.simiacryptus.mindseye.data.LabeledObject;

import de.javakaffee.kryoserializers.EnumMapSerializer;
import de.javakaffee.kryoserializers.EnumSetSerializer;
import de.javakaffee.kryoserializers.KryoReflectionFactorySupport;

public class Util {
  
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
          if (EnumSet.class.isAssignableFrom(clazz)) return new EnumSetSerializer();
          if (EnumMap.class.isAssignableFrom(clazz)) return new EnumMapSerializer();
          return super.getDefaultSerializer(clazz);
        }
        
      };
      return kryo;
    }
    
  };
  
  public static void add(final DoubleSupplier f, final double[] data) {
    for (int i = 0; i < data.length; i++)
    {
      data[i] += f.getAsDouble();
    }
  }
  
  public static Stream<byte[]> binaryStream(final String path, final String name, final int skip, final int recordSize) throws IOException {
    final DataInputStream in = new DataInputStream(new GZIPInputStream(new FileInputStream(new File(path, name))));
    in.skip(skip);
    return Util.toIterator(new BinaryChunkIterator(in, recordSize));
  }
  
  public static double bounds(final double value) {
    final int max = 0xFF;
    final int min = 0;
    return value < min ? min : value > max ? max : value;
  }
  
  public static double geomMean(final double[] error) {
    double sumLog = 0;
    for (final double element : error) {
      sumLog += Math.log(element);
    }
    return Math.exp(sumLog / error.length);
  }
  
  public static Kryo kryo() {
    return Util.threadKryo.get();
  }

  public static byte[] read(final DataInputStream i, final int s) throws IOException {
    final byte[] b = new byte[s];
    int pos = 0;
    while (b.length > pos) {
      final int read = i.read(b, pos, b.length - pos);
      if (0 == read) throw new RuntimeException();
      pos += read;
    }
    return b;
  }

  public static <T> List<T> shuffle(final List<T> buffer, final Random random) {
    final ArrayList<T> list = new ArrayList<T>(buffer);
    Collections.shuffle(list);
    return list;
  }

  public static NDArray toImage(final byte[] b) {
    final NDArray ndArray = new NDArray(28, 28);
    for (int x = 0; x < 28; x++)
    {
      for (int y = 0; y < 28; y++)
      {
        ndArray.set(new int[] { x, y }, b[x + y * 28]);
      }
    }
    return ndArray;
  }

  public static BufferedImage toImage(final NDArray ndArray) {
    final int[] dims = ndArray.getDims();
    final BufferedImage img = new BufferedImage(dims[0], dims[1], BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++)
    {
      for (int y = 0; y < img.getHeight(); y++)
      {
        if (ndArray.getDims()[2] == 1) {
          final double value = ndArray.get(x, y, 0);
          final int asByte = (int) Util.bounds(value) & 0xFF;
          img.setRGB(x, y, asByte * 0x010101);
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

  public static NDArray toNDArrayBW(final BufferedImage img) {
    final NDArray a = new NDArray(img.getWidth(), img.getHeight(), 1);
    for (int x = 0; x < img.getWidth(); x++)
    {
      for (int y = 0; y < img.getHeight(); y++)
      {
        a.set(new int[] { x, y, 0 }, img.getRGB(x, y) & 0xFF);
      }
    }
    return a;
  }

  public static NDArray toNDArrayRGB(final BufferedImage img) {
    final NDArray a = new NDArray(img.getWidth(), img.getHeight(), 3);
    for (int x = 0; x < img.getWidth(); x++)
    {
      for (int y = 0; y < img.getHeight(); y++)
      {
        a.set(new int[] { x, y, 0 }, img.getRGB(x, y) & 0xFF);
        a.set(new int[] { x, y, 1 }, img.getRGB(x, y) >> 8 & 0xFF);
        a.set(new int[] { x, y, 2 }, img.getRGB(x, y) >> 16 & 0x0FF);
      }
    }
    return a;
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
  
}
