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

import com.simiacryptus.util.ml.Tensor;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
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
    final Tensor inputSize = new Tensor(28, 28);

    public Network() {
      super();

      add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 2));
      add(new MaxSubsampleLayer(4, 4, 1));
      final Tensor[] input = { this.inputSize };
      add(new BiasLayer(eval(input).data[0].getDims()));
      add(new SigmoidActivationLayer());

      add(new ConvolutionSynapseLayer(new int[] { 2, 2, 2 }, 2));
      add(new MaxSubsampleLayer(2, 2, 1, 1));
      final Tensor[] input1 = { this.inputSize };
      add(new BiasLayer(eval(input1).data[0].getDims()));
      add(new SigmoidActivationLayer());
      final Tensor[] input2 = { this.inputSize };

      add(new DenseSynapseLayer(eval(input2).data[0].dim(), new int[] { 16 }));
      final Tensor[] input3 = { this.inputSize };
      add(new BiasLayer(eval(input3).data[0].getDims()));
      getChildren().add(new SigmoidActivationLayer());
      final Tensor[] input4 = { this.inputSize };

      add(new DenseSynapseLayer(eval(input4).data[0].dim(), new int[] { 10 }));
      final Tensor[] input5 = { this.inputSize };
      add(new BiasLayer(eval(input5).data[0].getDims()));
      add(new SoftmaxActivationLayer());
    }

  }

  private static final Logger log = LoggerFactory.getLogger(TestMNISTDev.class);

  public static final Random random = new Random();

  protected Network getNetwork() {
    return new Network();
  }

  private void report(final List<LabeledObject<Tensor>> buffer, final NNLayer<DAGNetwork> net) throws FileNotFoundException, IOException {
    final File outDir = new File("reports");
    outDir.mkdirs();
    final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
    final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
    final PrintStream out = new PrintStream(new FileOutputStream(report));
    out.println("<html><head></head><body>");
    buffer.stream().sorted(Comparator.comparing(img -> img.label))
        .map(x -> "<p>" + Util.toInlineImage(x.<BufferedImage>map(tensor -> tensor.toRgbImage())) + net.eval(x.data).data.toString() + "</p>").forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  @Test
  @Ignore
  public void test() throws Exception {
    TestMNISTDev.log.info("Starting");
    final NNLayer<DAGNetwork> net = getNetwork();
    final List<LabeledObject<Tensor>> buffer = TestMNISTDev.trainingDataStream().collect(Collectors.toList());
    final Tensor[][] data = TestMNISTDev.shuffle(buffer, TestMNISTDev.random).parallelStream().limit(1000).map(o -> new Tensor[] { o.data, TestMNISTDev.toOutNDArray(TestMNISTDev.toOut(o.label), 10) })
        .toArray(i2 -> new Tensor[i2][]);
    new Tester().init(data, net, new EntropyLossLayer()).setStaticRate(0.25).setVerbose(true).verifyConvergence(0.01, 1);
    report(buffer, net);
  }

  public static <T> List<T> shuffle(final List<T> buffer, final Random random) {
    final TreeMap<Double, T> tree = new TreeMap<Double, T>();
    if (!buffer.stream().allMatch(item -> null == tree.put(random.nextDouble(), item)))
      throw new RuntimeException();
    return tree.values().stream().collect(Collectors.toList());
  }

  public static Tensor fillImage(final byte[] b, final Tensor tensor) {
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        tensor.set(new int[] { x, y }, b[x + y * 28]&0xFF);
      }
    }
    return tensor;
  }

  public static Stream<LabeledObject<Tensor>> trainingDataStream() throws IOException {
    final String path = "C:/Users/Andrew Charneski/Downloads";
    final Stream<Tensor> imgStream = Util.binaryStream(path, "train-images-idx3-ubyte.gz", 16, 28 * 28).map(b->{
      return fillImage(b, new Tensor(28,28,1));
    });
    final Stream<byte[]> labelStream = Util.binaryStream(path, "train-labels-idx1-ubyte.gz", 8, 1);
  
    final Stream<LabeledObject<Tensor>> merged = Util.toStream(new Iterator<LabeledObject<Tensor>>() {
      Iterator<Tensor> imgItr = imgStream.iterator();
      Iterator<byte[]> labelItr = labelStream.iterator();
  
      @Override
      public boolean hasNext() {
        return this.imgItr.hasNext() && this.labelItr.hasNext();
      }
  
      @Override
      public LabeledObject<Tensor> next() {
        return new LabeledObject<Tensor>(this.imgItr.next(), Arrays.toString(this.labelItr.next()));
      }
    }, 100).limit(10000);
    return merged;
  }

  public static Tensor toOutNDArray(final int out, final int max) {
    final Tensor tensor = new Tensor(max);
    tensor.set(out, 1);
    return tensor;
  }

  public static int toOut(final String label) {
    for (int i = 0; i < 10; i++) {
      if (label.equals("[" + i + "]"))
        return i;
    }
    throw new RuntimeException();
  }

}
