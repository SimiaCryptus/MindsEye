package com.simiacryptus.mindseye.test.dev;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;

public class SimpleMNIST {
  private static final Logger log = LoggerFactory.getLogger(SimpleMNIST.class);

  public static int toOut(final String label) {
    for (int i = 0; i < 10; i++) {
      if (label.equals("[" + i + "]"))
        return i;
    }
    throw new RuntimeException();
  }

  public static NDArray toOutNDArray(final int out, final int max) {
    final NDArray ndArray = new NDArray(max);
    ndArray.set(out, 1);
    return ndArray;
  }

  final NDArray inputSize = new NDArray(28, 28);

  int verbose = 1;

  protected NNLayer<DAGNetwork> getNetwork() {
    final DAGNetwork net = new DAGNetwork();
    final NDArray[] input = { this.inputSize };
    net.add(new DenseSynapseLayer(net.eval(input).data.dim(), new int[] { 10 }));
    final NDArray[] input1 = { this.inputSize };
    net.add(new BiasLayer(net.eval(input1).data.getDims()));
    // net.add(new SigmoidActivationLayer().setVerbose(verbose));
    net.add(new SoftmaxActivationLayer());
    return net;
  }

  public Tester getTrainer(final NNLayer<DAGNetwork> net, final NDArray[][] data) {
    return new Tester().init(data, net, new EntropyLossLayer()).setVerbose(this.verbose > 0);
  }

  public Stream<LabeledObject<NDArray>> getTraining(final List<LabeledObject<NDArray>> buffer) {
    return Util.shuffle(buffer, Util.R.get()).parallelStream().limit(1000);
  }

  public Stream<LabeledObject<NDArray>> getVerification(final List<LabeledObject<NDArray>> buffer) {
    return buffer.parallelStream().limit(100);
  }

  @Test
  public void test() throws Exception {
    SimpleMNIST.log.info("Starting");
    final NNLayer<DAGNetwork> net = getNetwork();
    final List<LabeledObject<NDArray>> buffer = MNIST.trainingDataStream().collect(Collectors.toList());

    final NDArray[][] data = getTraining(buffer).map(o -> new NDArray[] { o.data, SimpleMNIST.toOutNDArray(SimpleMNIST.toOut(o.label), 10) }).toArray(i2 -> new NDArray[i2][]);

    getTrainer(net, data).verifyConvergence(0.01, 1);
    final double prevRms = getVerification(buffer).mapToDouble(o1 -> net.eval(o1.data).errMisclassification(SimpleMNIST.toOut(o1.label))).average().getAsDouble();
    SimpleMNIST.log.info("Tested RMS Error: {}", prevRms);
    MNIST.report(net);
  }

}
