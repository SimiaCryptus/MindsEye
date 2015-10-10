package com.simiacryptus.mindseye.test.demo.shapes;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.SynapseActivationLayer;
import com.simiacryptus.mindseye.test.Tester;

public class SoftmaxTests3 extends SimpleClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {

    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final int[] midSize = new int[] { 8 };
    final int midLayers = 0;
    DAGNetwork net = new DAGNetwork();

    final NNLayer<?> inputLayer = new DAGNetwork().add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)).add(new BiasLayer(midSize)).add(new SigmoidActivationLayer());
    net = net.add(inputLayer);

    for (int i = 0; i < midLayers; i++) {
      final NNLayer<?> hiddenLayer = new DAGNetwork().add(new DenseSynapseLayer(NDArray.dim(midSize), midSize)).add(new BiasLayer(midSize)).add(new SigmoidActivationLayer());
      net = net.add(hiddenLayer);
    }

    DAGNetwork outputLayer = new DAGNetwork();
    outputLayer = outputLayer.add(new SynapseActivationLayer(NDArray.dim(midSize)));
    outputLayer = outputLayer.add(new DenseSynapseLayer(NDArray.dim(midSize), outSize));
    outputLayer = outputLayer.add(new BiasLayer(outSize));
    outputLayer = outputLayer.add(new SynapseActivationLayer(NDArray.dim(outSize)));
    net = net.add(outputLayer);

    // outputLayer = outputLayer.add(new ExpActivationLayer());
    // outputLayer = outputLayer.add(new L1NormalizationLayer());
    // outputLayer = outputLayer.add(new SigmoidActivationLayer());
    outputLayer = outputLayer.add(new LinearActivationLayer());
    outputLayer = outputLayer.add(new SoftmaxActivationLayer());

    return net;
  }

  @Override
  public void test_Gaussians() throws Exception {
    super.test_Gaussians();
  }

  @Override
  public void test_II() throws Exception {
    super.test_II();
  }

  @Override
  public void test_III() throws Exception {
    super.test_III();
  }

  @Override
  public void test_Lines() throws Exception {

    super.test_Lines();
  }

  @Override
  public void test_O() throws Exception {
    super.test_O();
  }

  @Override
  public void test_O2() throws Exception {
    super.test_O2();
  }

  @Override
  public void test_O3() throws Exception {
    super.test_O3();
  }

  @Override
  public void test_oo() throws Exception {
    super.test_oo();
  }

  @Override
  public void test_simple() throws Exception {
    super.test_simple();
  }

  @Override
  public void test_snakes() throws Exception {
    super.test_snakes();
  }

  @Override
  public void test_sos() throws Exception {
    super.test_sos();
  }

  @Override
  public void test_X() throws Exception {
    super.test_X();
  }

  @Override
  public void test_xor() throws Exception {
    super.test_xor();
  }

  @Override
  public void verify(final Tester trainer) {
    // trainer.setVerbose(true).verifyConvergence(0, 0.0, 1);
    trainer.verifyConvergence(0.0, 5);
  }

}
