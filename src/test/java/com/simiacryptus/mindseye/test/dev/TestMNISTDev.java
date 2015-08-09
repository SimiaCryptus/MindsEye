package com.simiacryptus.mindseye.test.dev;

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
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
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

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.data.BinaryChunkIterator;
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.training.PipelineNetwork;

public class TestMNISTDev {
  public static class Network extends PipelineNetwork {
    final NDArray inputSize = new NDArray(28, 28);

    public Network() {
      super();

      add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 2));
      add(new MaxSubsampleLayer(4, 4, 1));
      add(new BiasLayer(eval(this.inputSize).data.getDims()));
      add(new SigmoidActivationLayer());

      add(new ConvolutionSynapseLayer(new int[] { 2, 2, 2 }, 2));
      add(new MaxSubsampleLayer(2, 2, 1, 1));
      add(new BiasLayer(eval(this.inputSize).data.getDims()));
      add(new SigmoidActivationLayer());
      
      this.layers.add(new DenseSynapseLayer(eval(this.inputSize).data.dim(), new int[] { 16 }));
      add(new BiasLayer(eval(this.inputSize).data.getDims()));
      this.layers.add(new SigmoidActivationLayer());

      add(new DenseSynapseLayer(eval(this.inputSize).data.dim(), new int[] { 10 }));
      add(new BiasLayer(eval(this.inputSize).data.getDims()));
      add(new SoftmaxActivationLayer());
    }

  }

  private static final Logger log = LoggerFactory.getLogger(TestMNISTDev.class);

  public static final Random random = new Random();

  public static Stream<byte[]> binaryStream(final String path, final String name, final int skip, final int recordSize) throws IOException {
    final DataInputStream in = new DataInputStream(new GZIPInputStream(new FileInputStream(new File(path, name))));
    in.skip(skip);
    return TestMNISTDev.toIterator(new BinaryChunkIterator(in, recordSize));
  }

  private static double bounds(final double value) {
    return value < 0 ? 0 : value > 0xFF ? 0xFF : value;
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
          final int asByte = (int) TestMNISTDev.bounds(value) & 0xFF;
          img.setRGB(x, y, asByte * 0x010101);
        } else {
          final double red = TestMNISTDev.bounds(ndArray.get(x, y, 0));
          final double green = TestMNISTDev.bounds(ndArray.get(x, y, 1));
          final double blue = TestMNISTDev.bounds(ndArray.get(x, y, 2));
          img.setRGB(x, y, (int) (red + ((int) green << 8) + ((int) blue << 16)));
        }
      }
    }
    return img;
  }
  
  public static String toInlineImage(final BufferedImage img, final String alt) {
    return TestMNISTDev.toInlineImage(new LabeledObject<BufferedImage>(img, alt));
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

  public static NDArray toNDArray1(final BufferedImage img) {
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

  public static NDArray toNDArray3(final BufferedImage img) {
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

  public static int toOut(final String label) {
    for (int i = 0; i < 10; i++)
    {
      if (label.equals("[" + i + "]")) return i;
    }
    throw new RuntimeException();
  }

  public static NDArray toOutNDArray(final int out, final int max) {
    final NDArray ndArray = new NDArray(max);
    ndArray.set(out, 1);
    return ndArray;
  }
  
  public static Stream<LabeledObject<NDArray>> trainingDataStream() throws IOException {
    final String path = "C:/Users/Andrew Charneski/Downloads";
    final Stream<NDArray> imgStream = TestMNISTDev.binaryStream(path, "train-images-idx3-ubyte.gz", 16, 28 * 28).map(TestMNISTDev::toImage);
    final Stream<byte[]> labelStream = TestMNISTDev.binaryStream(path, "train-labels-idx1-ubyte.gz", 8, 1);

    final Stream<LabeledObject<NDArray>> merged = Util.toStream(new Iterator<LabeledObject<NDArray>>() {
      Iterator<NDArray> imgItr = imgStream.iterator();
      Iterator<byte[]> labelItr = labelStream.iterator();

      @Override
      public boolean hasNext() {
        return this.imgItr.hasNext() && this.labelItr.hasNext();
      }

      @Override
      public LabeledObject<NDArray> next() {
        return new LabeledObject<NDArray>(this.imgItr.next(), Arrays.toString(this.labelItr.next()));
      }
    }, 100).limit(10000);
    return merged;
  }

  protected Network getNetwork() {
    return new Network();
  }

  private void report(final List<LabeledObject<NDArray>> buffer, final PipelineNetwork net) throws FileNotFoundException, IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    buffer.stream()
    .sorted(Comparator.comparing(img -> img.label))
    .map(x -> "<p>" + TestMNISTDev.toInlineImage(x.<BufferedImage> map(TestMNISTDev::toImage)) + net.eval(x.data).data.toString() + "</p>")
    .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  @Test
  public void test() throws Exception {
    TestMNISTDev.log.info("Starting");
    final PipelineNetwork net = getNetwork()
        .setMutationAmplitude(2)
        ;
    final List<LabeledObject<NDArray>> buffer = TestMNISTDev.trainingDataStream().collect(Collectors.toList());
    final NDArray[][] data = Util.shuffle(buffer, TestMNISTDev.random).parallelStream().limit(1000)
        .map(o -> new NDArray[] { o.data, TestMNISTDev.toOutNDArray(TestMNISTDev.toOut(o.label), 10) })
        .toArray(i2 -> new NDArray[i2][]);
    net.trainer(data)
        .setMutationAmount(.1)
        //.setStaticRate(0.2)
        .setVerbose(true)
        .verifyConvergence(10000, 0.01, 1);
    {
      final PipelineNetwork net2 = net;
      final double prevRms = buffer.parallelStream().limit(100).mapToDouble(o1 -> net2.eval(o1.data).errMisclassification(TestMNISTDev.toOut(o1.label)))
          .average().getAsDouble();
      TestMNISTDev.log.info("Tested RMS Error: {}", prevRms);
    }
    report(buffer, net);
  }

}
