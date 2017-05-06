package com.simiacryptus.mindseye.test.demo.shapes;

import java.util.function.BiFunction;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;

public class SoftmaxTests2 extends SimpleClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {

    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final int[] midSize = new int[] { 10 };
    final int midLayers = 0;
    DAGNetwork net = new PipelineNetwork();

    // net.add(new
    // SynapseActivationLayer(Tensor.dim(inputSize)).setWeights(()->1.));
    net.add(new DenseSynapseLayer(Tensor.dim(inputSize), midSize));
    net.add(new BiasLayer(midSize));
    // net.add(new LinearActivationLayer());
    // net.add(new
    // SynapseActivationLayer(Tensor.dim(midSize)).setWeights(()->1.));
    net.add(new SigmoidActivationLayer());

    for (int i = 0; i < midLayers; i++) {
      net.add(new DenseSynapseLayer(Tensor.dim(midSize), midSize));
      net.add(new BiasLayer(midSize));
      // net.add(new LinearActivationLayer());
      net.add(new SigmoidActivationLayer());
    }

    // net.add(new
    // SynapseActivationLayer(Tensor.dim(midSize)).setWeights(()->1.));
    net.add(new DenseSynapseLayer(Tensor.dim(midSize), outSize));
    // net.add(new PermutationLayer());
    // net.add(new
    // SynapseActivationLayer(Tensor.dim(outSize)).setWeights(()->1.));
    net.add(new BiasLayer(outSize));

    // net.add(new ExpActivationLayer());
    // net.add(new L1NormalizationLayer());
    // net.add(new LinearActivationLayer());
    // net.add(new SigmoidActivationLayer());
    net.add(new SoftmaxActivationLayer());

    return net;
  }

  @Override
  public Tester buildTrainer(final Tensor[][] samples, final NNLayer<DAGNetwork> net) {
    return new Tester().init(samples, net, new EntropyLossLayer());
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
  public void test_O22() throws Exception {
    super.test_O22();
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
  public void train(final NNLayer<DAGNetwork> net, final Tensor[][] trainingsamples, final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler) {
    final Tester trainer = buildTrainer(trainingsamples, net);
    trainer.handler.add(resultHandler);
    trainer.setVerbose(true);
    trainer.verifyConvergence(0.01, 1);
  }

  @Override
  public int height() {
    return (int) (.25*super.height());
  }

  @Override
  public int width() {
    return (int) (.25*super.width());
  }

}
