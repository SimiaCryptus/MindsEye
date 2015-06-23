package com.simiacryptus.mindseye;

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
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

import javax.imageio.ImageIO;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.input.BoundedInputStream;
import org.apache.commons.io.output.ByteArrayOutputStream;
import org.junit.Test;

public class TestCIFAR {
  
  @Test
  public void test() throws Exception {
    
    File outDir = new File("reports");
    outDir.mkdirs();
    File report = new File(outDir, this.getClass().getSimpleName()+".html");
    PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    
    String path = "C:/Users/Andrew Charneski/Downloads";
    String name = "cifar-10-binary.tar.gz";
    tarStream(path, name)
        .flatMap(in -> BinaryChunkIterator.toStream(new BinaryChunkIterator(new DataInputStream(in), 3073)))
        .map(this::toImage)
        .sorted(Comparator.comparing(img->img.name))
        .map(this::toInlineImage)
        .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  private Stream<BoundedInputStream> tarStream(String path, String name) throws IOException {
    return BinaryChunkIterator.toStream(new Iterator<BoundedInputStream>() {
      FileInputStream f = new FileInputStream(new File(path, name));
      TarArchiveInputStream tar = new TarArchiveInputStream(new GZIPInputStream(f));
      
      @Override
      public boolean hasNext() {
        try {
          int available = f.available();
          return 0 < available;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
      
      @Override
      public BoundedInputStream next() {
        TarArchiveEntry nextTarEntry;
        try {
          nextTarEntry = tar.getNextTarEntry();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
        return new BoundedInputStream(tar, nextTarEntry.getSize());
      }
    });
  }
  
  public LabeledImage toImage(byte[] b) {
    BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++)
    {
      for (int y = 0; y < img.getHeight(); y++)
      {
        int red = 0xFF & b[1 + 1024 * 0 + x + y * 32];
        int blue = 0xFF & b[1 + 1024 * 1 + x + y * 32];
        int green = 0xFF & b[1 + 1024 * 2 + x + y * 32];
        int c = (red << 16) + (blue << 8) + green;
        img.setRGB(x, y, c);
      }
    }
    return new LabeledImage(img, Arrays.toString(new byte[]{b[0]}));
  }
  
  public String toInlineImage(LabeledImage img) {
    ByteArrayOutputStream b = new ByteArrayOutputStream();
    try {
      ImageIO.write(img.img, "PNG", b);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    byte[] byteArray = b.toByteArray();
    String encode = Base64.getEncoder().encodeToString(byteArray);
    return "<img src=\"data:image/png;base64," + encode + "\" alt=\""+img.name+"\" />";
  }
}
