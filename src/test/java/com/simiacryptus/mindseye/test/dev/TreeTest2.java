package com.simiacryptus.mindseye.test.dev;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.TreeNetwork;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.demo.shapes.SimpleClassificationTests;

public class TreeTest2 extends SimpleClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {

    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };

    final NNLayer<DAGNetwork> net = new TreeNetwork(inputSize, outSize) {
    
      /**
       * 
       */
      private static final long serialVersionUID = 8233151944966481700L;
    
      @Override
      public NNLayer<DAGNetwork> buildGate() {
        DAGNetwork gate = new DAGNetwork();
        final int mid = 10;
        gate = gate.add(new DenseSynapseLayer(NDArray.dim(this.inputSize), new int[] { mid }).setWeights(() -> Util.R.get().nextGaussian()));
        // gate = gate.add(new DenseSynapseLayer(mid, new int[] { mid
        // }).setWeights(()->Util.R.get().nextGaussian()));
        gate = gate.add(new BiasLayer(new int[] { mid }));
        gate = gate.add(new SigmoidActivationLayer());
        gate = gate.add(new DenseSynapseLayer(mid, this.outSize).setWeights(() -> Util.R.get().nextGaussian()));
        gate = gate.add(new BiasLayer(this.outSize));
        gate = gate.add(new SoftmaxActivationLayer());
        return gate;
      }
    
    };
    // net = net.add(new MinMaxFilterLayer());
    // net = net.add(new SigmoidActivationLayer());
    return net;
  }

  @Override
  public Tester buildTrainer(final NDArray[][] samples, final NNLayer<DAGNetwork> net) {
    return new Tester().init(samples, net, new EntropyLossLayer());
    // return net.trainer(samples, new MaxEntropyLossLayer());
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
  public void verify(final Tester trainer) {
    trainer.setVerbose(true);
    // trainer.getInner().getDynamicRateTrainer().setStopError(-Double.POSITIVE_INFINITY);
    // trainer.getInner().setAlignEnabled(false);
    // trainer.verifyConvergence(-Double.POSITIVE_INFINITY, 1);
    trainer.verifyConvergence(0.01, 10);
  }

}
