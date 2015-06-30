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
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.data.BinaryChunkIterator;
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;

public class TestNetworkDev {
  public static final class Network extends PipelineNetwork {
    
    public Network() {
      super();
      final NDArray inputSize = new NDArray(28, 28);
      
      // layers.add(new NormalizerLayer(inputSize.getDims()));
      
      add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 3)
          .fillWeights(() -> 0.001 * TestNetworkDev.random.nextGaussian()));
      add(new MaxSubsampleLayer(4, 4, 1));
      add(new BiasLayer(eval(inputSize).data.getDims()));
      add(new SigmoidActivationLayer());
      
      add(new ConvolutionSynapseLayer(new int[] { 2, 2, 2 }, 2)
          .fillWeights(() -> 0.001 * TestNetworkDev.random.nextGaussian()));
      add(new MaxSubsampleLayer(2, 2, 1, 1));
      add(new BiasLayer(eval(inputSize).data.getDims()));
      add(new SigmoidActivationLayer());
      
      // layers.add(new DenseSynapseLayer(eval(inputSize).data.dim(), new int[] { 16 })
      // .fillWeights(() -> 0.001 * random.nextGaussian()));
      // layers.add(new SigmoidActivationLayer());
      
      add(new DenseSynapseLayer(eval(inputSize).data.dim(), new int[] { 10 })
          .addWeights(() -> 0.001 * TestNetworkDev.random.nextGaussian()));
      add(new BiasLayer(eval(inputSize).data.getDims()));
      // layers.add(new BiasLayer(eval(inputSize).data.getDims()));
      add(new SigmoidActivationLayer());
      // add(new SoftmaxActivationLayer());
    }
    
  }
  
  private static final Logger log = LoggerFactory.getLogger(TestNetworkDev.class);
  
  public static final Random random = new Random();
  
  public static Stream<byte[]> binaryStream(final String path, final String name, final int skip, final int recordSize) throws IOException {
    final DataInputStream in = new DataInputStream(new GZIPInputStream(new FileInputStream(new File(path, name))));
    in.skip(skip);
    return TestNetworkDev.toIterator(new BinaryChunkIterator(in, recordSize));
  }
  
  public static <T> Stream<T> toIterator(final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }
  
  private void report(final List<LabeledObject<NDArray>> buffer) throws FileNotFoundException, IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    buffer.stream()
        .sorted(Comparator.comparing(img -> img.label))
        .map(x -> x.<BufferedImage> map(this::toImage))
        .map(this::toInlineImage)
        .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }
  
  @Test
  public void test() throws Exception {
    TestNetworkDev.log.info("Starting");
    
    final NNLayer net = new Network();
    
    final List<LabeledObject<NDArray>> buffer = trainingDataStream().collect(Collectors.toList());
    double prevRms = buffer.parallelStream().limit(100).mapToDouble(o1 -> net.eval(o1.data).errMisclassification(toOut(o1.label))).average().getAsDouble();
    TestNetworkDev.log.info("Initial RMS Error: {}", prevRms);
    double learningRate = 0.1;
    final double adaptivity = 1.1;
    final double decayBias = 1.5;
    for (int i = 0; i < 1000; i++)
    {
      if (i < 5) {
        learningRate = 0.2;
      } else if (i < 50) {
        learningRate = 0.01;
      } else {
        learningRate = 0.0001;
      }
      
      final double currentRate = learningRate;
      Util.shuffle(buffer, TestNetworkDev.random).parallelStream().limit(100).forEach(o -> {
        net.eval(o.data).learn(currentRate, toOut(o.label));
      });
      final double rms = Util.shuffle(buffer, TestNetworkDev.random).parallelStream().limit(10).mapToDouble(o1 -> net.eval(o1.data).errRms(toOut(o1.label)))
          .average()
          .getAsDouble();
      TestNetworkDev.log.info("RMS Error: {}; Learning Rate: {}", rms, currentRate);
      if (rms < prevRms) {
        learningRate *= adaptivity;
      } else {
        learningRate /= Math.pow(adaptivity, decayBias);
      }
      prevRms = rms;
    }
    
    report(buffer);
  }
  
  public NDArray toImage(final byte[] b) {
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
  
  public BufferedImage toImage(final NDArray ndArray) {
    final BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < 28; x++)
    {
      for (int y = 0; y < 28; y++)
      {
        img.setRGB(x, y, (int) ndArray.get(x, y) * 0x00010101);
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
  
  private int toOut(final String label) {
    for (int i = 0; i < 10; i++)
    {
      if (label.equals("[" + i + "]")) return i;
    }
    throw new RuntimeException();
  }
  
  private Stream<LabeledObject<NDArray>> trainingDataStream() throws IOException {
    final String path = "C:/Users/Andrew Charneski/Downloads";
    final Stream<NDArray> imgStream = TestNetworkDev.binaryStream(path, "train-images-idx3-ubyte.gz", 16, 28 * 28).map(this::toImage);
    final Stream<byte[]> labelStream = TestNetworkDev.binaryStream(path, "train-labels-idx1-ubyte.gz", 8, 1);
    
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
  
}
