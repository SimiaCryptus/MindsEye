package com.simiacryptus.mindseye.test.dev;

import java.awt.Desktop;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;

public class TestMNISTDev {
  public static class Network extends DAGNetwork {
    /**
     * 
     */
    private static final long serialVersionUID = -2958566888774127417L;
    final NDArray inputSize = new NDArray(28, 28);

    public Network() {
      super();

      add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 2));
      add(new MaxSubsampleLayer(4, 4, 1));
      final NDArray[] input = { this.inputSize };
      add(new BiasLayer(eval(input).data[0].getDims()));
      add(new SigmoidActivationLayer());

      add(new ConvolutionSynapseLayer(new int[] { 2, 2, 2 }, 2));
      add(new MaxSubsampleLayer(2, 2, 1, 1));
      final NDArray[] input1 = { this.inputSize };
      add(new BiasLayer(eval(input1).data[0].getDims()));
      add(new SigmoidActivationLayer());
      final NDArray[] input2 = { this.inputSize };

      add(new DenseSynapseLayer(eval(input2).data[0].dim(), new int[] { 16 }));
      final NDArray[] input3 = { this.inputSize };
      add(new BiasLayer(eval(input3).data[0].getDims()));
      getChildren().add(new SigmoidActivationLayer());
      final NDArray[] input4 = { this.inputSize };

      add(new DenseSynapseLayer(eval(input4).data[0].dim(), new int[] { 10 }));
      final NDArray[] input5 = { this.inputSize };
      add(new BiasLayer(eval(input5).data[0].getDims()));
      add(new SoftmaxActivationLayer());
    }

  }

  private static final Logger log = LoggerFactory.getLogger(TestMNISTDev.class);

  public static final Random random = new Random();

  protected Network getNetwork() {
    return new Network();
  }

  private void report(final List<LabeledObject<NDArray>> buffer, final NNLayer<DAGNetwork> net) throws FileNotFoundException, IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    buffer.stream().sorted(Comparator.comparing(img -> img.label))
        .map(x -> "<p>" + Util.toInlineImage(x.<BufferedImage>map(Util::toImage)) + net.eval(x.data).data.toString() + "</p>").forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  @Test
  public void test() throws Exception {
    TestMNISTDev.log.info("Starting");
    final NNLayer<DAGNetwork> net = getNetwork();
    final List<LabeledObject<NDArray>> buffer = TestMNISTDev.trainingDataStream().collect(Collectors.toList());
    final NDArray[][] data = TestMNISTDev.shuffle(buffer, TestMNISTDev.random).parallelStream().limit(1000).map(o -> new NDArray[] { o.data, TestMNISTDev.toOutNDArray(TestMNISTDev.toOut(o.label), 10) })
        .toArray(i2 -> new NDArray[i2][]);
    new Tester().init(data, net, new EntropyLossLayer()).setStaticRate(0.25).setVerbose(true).verifyConvergence(0.01, 1);
    report(buffer, net);
  }

  public static <T> List<T> shuffle(final List<T> buffer, final Random random) {
    final TreeMap<Double, T> tree = new TreeMap<Double, T>();
    if (!buffer.stream().allMatch(item -> null == tree.put(random.nextDouble(), item)))
      throw new RuntimeException();
    return tree.values().stream().collect(Collectors.toList());
  }

  public static Stream<LabeledObject<NDArray>> trainingDataStream() throws IOException {
    final String path = "C:/Users/Andrew Charneski/Downloads";
    final Stream<NDArray> imgStream = Util.binaryStream(path, "train-images-idx3-ubyte.gz", 16, 28 * 28).map(b->{
      return Util.fillImage(b, new NDArray(28,28,1));
    });
    final Stream<byte[]> labelStream = Util.binaryStream(path, "train-labels-idx1-ubyte.gz", 8, 1);
  
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

  public static NDArray toOutNDArray(final int out, final int max) {
    final NDArray ndArray = new NDArray(max);
    ndArray.set(out, 1);
    return ndArray;
  }

  public static int toOut(final String label) {
    for (int i = 0; i < 10; i++) {
      if (label.equals("[" + i + "]"))
        return i;
    }
    throw new RuntimeException();
  }

}
