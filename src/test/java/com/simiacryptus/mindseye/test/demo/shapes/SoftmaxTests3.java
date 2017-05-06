package com.simiacryptus.mindseye.test.demo.shapes;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;

public class SoftmaxTests3 extends SimpleClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {

    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final int[] midSize = new int[] { 8 };
    final int midLayers = 0;
    DAGNetwork net = new PipelineNetwork();

    PipelineNetwork inputLayer = new PipelineNetwork();
    inputLayer.add(new DenseSynapseLayer(Tensor.dim(inputSize), midSize));
    inputLayer.add(new BiasLayer(midSize));
    inputLayer.add(new SigmoidActivationLayer());
    net.add(inputLayer);

    for (int i = 0; i < midLayers; i++) {
      final PipelineNetwork hiddenLayer = new PipelineNetwork();
      hiddenLayer.add(new DenseSynapseLayer(Tensor.dim(midSize), midSize));
      hiddenLayer.add(new BiasLayer(midSize));
      hiddenLayer.add(new SigmoidActivationLayer());
      net.add(hiddenLayer);
    }

    DAGNetwork outputLayer = new PipelineNetwork();
    outputLayer.add(new DenseSynapseLayer(Tensor.dim(midSize), outSize));
    outputLayer.add(new BiasLayer(outSize));
    net.add(outputLayer);

    // outputLayer = outputLayer.add(new ExpActivationLayer());
    // outputLayer = outputLayer.add(new L1NormalizationLayer());
    // outputLayer = outputLayer.add(new SigmoidActivationLayer());
    outputLayer.add(new LinearActivationLayer());
    outputLayer.add(new SoftmaxActivationLayer());

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

//  @Override
//  public void verify(final Tester trainer) {
//    // trainer.setVerbose(true).verifyConvergence(0, 0.0, 1);
//    trainer.verifyConvergence(0.0, 5);
//  }

}
