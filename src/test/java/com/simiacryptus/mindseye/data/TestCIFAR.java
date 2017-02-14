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
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

import javax.imageio.ImageIO;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.input.BoundedInputStream;
import org.apache.commons.io.output.ByteArrayOutputStream;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.BinaryChunkIterator;
import com.simiacryptus.mindseye.core.LabeledObject;

public class TestCIFAR {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TestCIFAR.class);

  private Stream<BoundedInputStream> tarStream(final String path, final String name) throws IOException {
    return Util.toStream(new Iterator<BoundedInputStream>() {
      FileInputStream f = new FileInputStream(new File(path, name));
      TarArchiveInputStream tar = new TarArchiveInputStream(new GZIPInputStream(this.f));

      @Override
      public boolean hasNext() {
        try {
          final int available = this.f.available();
          return 0 < available;
        } catch (final IOException e) {
          throw new RuntimeException(e);
        }
      }

      @Override
      public BoundedInputStream next() {
        TarArchiveEntry nextTarEntry;
        try {
          nextTarEntry = this.tar.getNextTarEntry();
        } catch (final IOException e) {
          throw new RuntimeException(e);
        }
        return new BoundedInputStream(this.tar, nextTarEntry.getSize());
      }
    });
  }

  @Test
  @Ignore
  public void test() throws Exception {

    final File outDir = new File("reports");
    outDir.mkdirs();
    final File report = new File(outDir, this.getClass().getSimpleName() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");

    final String path = "C:/Users/Andrew Charneski/Downloads";
    final String name = "cifar-10-binary.tar.gz";
    tarStream(path, name).flatMap(in -> Util.toStream(new BinaryChunkIterator(new DataInputStream(in), 3073))).map(this::toImage).sorted(Comparator.comparing(img -> img.label))
        .map(this::toInlineImage).forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  public LabeledObject<BufferedImage> toImage(final byte[] b) {
    final BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        final int red = 0xFF & b[1 + 1024 * 0 + x + y * 32];
        final int blue = 0xFF & b[1 + 1024 * 1 + x + y * 32];
        final int green = 0xFF & b[1 + 1024 * 2 + x + y * 32];
        final int c = (red << 16) + (blue << 8) + green;
        img.setRGB(x, y, c);
      }
    }
    return new LabeledObject<BufferedImage>(img, Arrays.toString(new byte[] { b[0] }));
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
