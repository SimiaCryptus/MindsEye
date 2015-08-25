package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;

public class MonteCarloClassificationSoftmaxNetworkTests1 extends MonteCarloClassificationSoftmaxNetworkTests {
  
  @Override
  public PipelineNetwork buildNetwork() {
    
    final int[] midSize = new int[] { 4 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    PipelineNetwork net = new PipelineNetwork()
      
      // .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
      
      .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize))
      .add(new BiasLayer(midSize))
      .add(new SigmoidActivationLayer())
      
//      .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
//      .add(new BiasLayer(midSize))
//      .add(new SigmoidActivationLayer())
//            
//      .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
//      .add(new BiasLayer(midSize))
//      .add(new SigmoidActivationLayer())

      // .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
      // .add(new BiasLayer(midSize))
      // .add(new SigmoidActivationLayer())
      
      .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
      .add(new BiasLayer(outSize));
      
      // .add(new SigmoidActivationLayer());
      //.add(new SoftmaxActivationLayer().setVerbose(false));
    return net;
  }
  
  @Override
  public void verify(Trainer trainer) {
    trainer.verifyConvergence(0, 0.0, 10);
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
  
}
