package com.simiacryptus.mindseye.data;

import java.awt.Desktop;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Base64;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

import javax.imageio.ImageIO;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.util.BinaryChunkIterator;
import com.simiacryptus.mindseye.util.LabeledObject;
import com.simiacryptus.mindseye.util.Util;

public class TestMNIST {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TestCIFAR.class);
  
  public static Stream<byte[]> binaryStream(final String path, final String name, final int skip, final int recordSize) throws IOException {
    final DataInputStream in = new DataInputStream(new GZIPInputStream(new FileInputStream(new File(path, name))));
    in.skip(skip);
    return TestMNIST.toIterator(new BinaryChunkIterator(in, recordSize));
  }
  
  public static <T> Stream<T> toIterator(final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }
  
  @Test
  public void test() throws Exception {
    
    final File outDir = new File("reports");
    outDir.mkdirs();
    final File report = new File(outDir, this.getClass().getSimpleName() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    final String path = "C:/Users/Andrew Charneski/Downloads";
    out.println("<html><head></head><body>");
    final Stream<BufferedImage> imgStream = TestMNIST.binaryStream(path, "train-images-idx3-ubyte.gz", 16, 28 * 28).map(this::toImage);
    final Stream<byte[]> labelStream = TestMNIST.binaryStream(path, "train-labels-idx1-ubyte.gz", 8, 1);
    
    final Stream<LabeledObject<BufferedImage>> merged = Util.toStream(new Iterator<LabeledObject<BufferedImage>>() {
      Iterator<BufferedImage> imgItr = imgStream.iterator();
      Iterator<byte[]> labelItr = labelStream.iterator();
      
      @Override
      public boolean hasNext() {
        return this.imgItr.hasNext() && this.labelItr.hasNext();
      }
      
      @Override
      public LabeledObject<BufferedImage> next() {
        return new LabeledObject<BufferedImage>(this.imgItr.next(), Arrays.toString(this.labelItr.next()));
      }
    }, 100);
    
    merged
        .collect(Collectors.toList())
        .stream()
        .sorted(Comparator.comparing(img -> img.label))
        .map(this::toInlineImage)
        .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }
  
  public BufferedImage toImage(final byte[] b) {
    final BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        img.setRGB(x, y, b[x + y * 28] * 0x00010101);
      }
    }
    return img;
  }
  
  public String toInlineImage(final LabeledObject<BufferedImage> img) {
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
  
}
