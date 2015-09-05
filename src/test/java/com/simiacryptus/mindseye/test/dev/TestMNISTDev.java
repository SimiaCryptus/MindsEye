package com.simiacryptus.mindseye.test.dev;

import java.awt.Desktop;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.util.LabeledObject;
import com.simiacryptus.mindseye.util.Util;

public class TestMNISTDev {
  public static class Network extends PipelineNetwork {
    final NDArray inputSize = new NDArray(28, 28);

    public Network() {
      super();

      add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 2));
      add(new MaxSubsampleLayer(4, 4, 1));
      NDArray[] input = { this.inputSize };
      add(new BiasLayer(eval(input).data.getDims()));
      add(new SigmoidActivationLayer());

      add(new ConvolutionSynapseLayer(new int[] { 2, 2, 2 }, 2));
      add(new MaxSubsampleLayer(2, 2, 1, 1));
      NDArray[] input1 = { this.inputSize };
      add(new BiasLayer(eval(input1).data.getDims()));
      add(new SigmoidActivationLayer());
      NDArray[] input2 = { this.inputSize };
      
      add(new DenseSynapseLayer(eval(input2).data.dim(), new int[] { 16 }));
      NDArray[] input3 = { this.inputSize };
      add(new BiasLayer(eval(input3).data.getDims()));
      this.getChildren().add(new SigmoidActivationLayer());
      NDArray[] input4 = { this.inputSize };

      add(new DenseSynapseLayer(eval(input4).data.dim(), new int[] { 10 }));
      NDArray[] input5 = { this.inputSize };
      add(new BiasLayer(eval(input5).data.getDims()));
      add(new SoftmaxActivationLayer());
    }

  }

  private static final Logger log = LoggerFactory.getLogger(TestMNISTDev.class);

  public static final Random random = new Random();

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
    buffer.stream().sorted(Comparator.comparing(img -> img.label))
    .map(x -> "<p>" + Util.toInlineImage(x.<BufferedImage> map(Util::toImage)) + net.eval(x.data).data.toString() + "</p>")
    .forEach(out::println);
    out.println("</body></html>");
    out.close();
    Desktop.getDesktop().browse(report.toURI());
  }

  @Test
  public void test() throws Exception {
    TestMNISTDev.log.info("Starting");
    final PipelineNetwork net = getNetwork();
    final List<LabeledObject<NDArray>> buffer = Util.trainingDataStream().collect(Collectors.toList());
    final NDArray[][] data = Util.shuffle(buffer, TestMNISTDev.random).parallelStream().limit(1000)
        .map(o -> new NDArray[] { o.data, Util.toOutNDArray(Util.toOut(o.label), 10) })
        .toArray(i2 -> new NDArray[i2][]);
    net.trainer(data)
    .setMutationAmplitude(2)
    .setMutationAmount(.1)
    .setStaticRate(0.25)
    .setVerbose(true).verifyConvergence(10000, 0.01, 1);
    {
      final PipelineNetwork net2 = net;
      final double prevRms = buffer.parallelStream().limit(100).mapToDouble(o1 -> net2.eval(o1.data).errMisclassification(Util.toOut(o1.label)))
          .average().getAsDouble();
      TestMNISTDev.log.info("Tested RMS Error: {}", prevRms);
    }
    report(buffer, net);
  }

}
