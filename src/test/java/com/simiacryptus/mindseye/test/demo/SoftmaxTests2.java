package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;

public class SoftmaxTests2 extends SimpleClassificationTests {

  @Override
  public PipelineNetwork buildNetwork() {

    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final int[] midSize = new int[] { 5 };
    final int midLayers = 1;
    PipelineNetwork net = new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)
    {
      // @Override
      // protected double getMobility() {
      // // //if(true) return 1;
      // double status = getStatus();
      // double x = (0.5 - status) * 5;
      // double sigmiod = SigmoidActivationLayer.sigmiod(x);
      // return sigmiod;
      // }
    }
        )
        .add(new BiasLayer(midSize)
        {
          // @Override
          // protected double getMobility() {
          // // //if(true) return 1;
          // double status = getStatus();
          // double x = (0.7 - status) * 5;
          // double sigmiod = SigmoidActivationLayer.sigmiod(x);
          // return sigmiod;
          // }
        }
            )
            .add(new SigmoidActivationLayer()
            {
              // @Override
              // protected double getNonlinearity() {
              // // if(true) return 1;
              // double status = getStatus();
              // double sigmiod = SigmoidActivationLayer.sigmiod((0.8 - status) * 2);
              // double sigmiod2 = SigmoidActivationLayer.sigmiod((0.6 - status) * 5);
              // return (sigmiod+sigmiod2)/2;
              // }
            }
                );

    for (int i = 0; i < midLayers; i++) {
      net = net.add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
          .add(new BiasLayer(midSize))
          .add(new SigmoidActivationLayer());
    }

    net = net.add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize));

    net = net.add(new SigmoidActivationLayer());

    // net = net.add(new SoftmaxActivationLayer() {
    // @Override
    // protected double getNonlinearity() {
    // // return getStatus()<.2?1:1e-5;
    // // if(true) return 0.;
    // double status = getStatus();
    // double x = (0.3 - status) * 15;
    // return SigmoidActivationLayer.sigmiod(x);
    // }
    // }.setVerbose(false));

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
  public void verify(final Trainer trainer) {
    trainer.verifyConvergence(0, 0.01, 10);
  }

}
