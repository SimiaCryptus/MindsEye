package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.Graphics2D;
import java.io.IOException;
import java.util.Iterator;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.regression.MNISTClassificationTest;

public class SimpliedConvolutionLearningTest extends MNISTClassificationTest {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    DAGNetwork net = new DAGNetwork();
    final int n = 2;
    final int m = 28 - n + 1;
    net = net.add(new ConvolutionSynapseLayer(new int[] { n, n }, 10).addWeights(() -> Util.R.get().nextGaussian() * .001));
    net = net.add(new SqActivationLayer());
    net = net.add(new SumSubsampleLayer(new int[] { m, m, 1 }));
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }

  @Override
  public Tester buildTrainer(final Tensor[][] samples, final NNLayer<DAGNetwork> net) {
    final EntropyLossLayer lossLayer = new EntropyLossLayer();
    final Tester trainer = new Tester().setMaxDynamicRate(1000).init(samples, net, lossLayer);
    trainer.setVerbose(true);
    final TrainingContext trainingContext = trainer.trainingContext();
    trainingContext.setTimeout(5, java.util.concurrent.TimeUnit.MINUTES);
    return trainer;
  }

  public Stream<LabeledObject<Tensor>> getTrainingData() throws IOException {

    final Stream<LabeledObject<Tensor>> merged = Util.toStream(new Iterator<LabeledObject<Tensor>>() {
      int cnt = 0;

      @Override
      public boolean hasNext() {
        return true;
      }

      @Override
      public LabeledObject<Tensor> next() {
        final int index = this.cnt++;
        String id = "";
        Tensor imgData;
        while (true) {
          final java.awt.image.BufferedImage img = new java.awt.image.BufferedImage(28, 28, java.awt.image.BufferedImage.TYPE_BYTE_GRAY);
          final Graphics2D g = img.createGraphics();
          final int cardinality = index % 2;
          if (0 == cardinality) {
            final int x1 = Util.R.get().nextInt(28);
            final int x2 = Util.R.get().nextInt(28);
            final int y = Util.R.get().nextInt(28);
            if (Math.abs(x1 - x2) < 1) {
              continue;
            }
            g.drawLine(x1, y, x2, y);
            id = "[0]";
          } else if (1 == cardinality) {
            final int x = Util.R.get().nextInt(28);
            final int y1 = Util.R.get().nextInt(28);
            final int y2 = Util.R.get().nextInt(28);
            if (Math.abs(y1 - y2) < 1) {
              continue;
            }
            g.drawLine(x, y1, x, y2);
            id = "[1]";
          }
          imgData = new Tensor(new int[] { 28, 28, 1 }, img.getData().getSamples(0, 0, 28, 28, 0, (double[]) null));
          break;
        }
        return new LabeledObject<Tensor>(imgData, id);
      }
    }, 1000).limit(1000);
    return merged;
  }

  @Override
  public int height() {
    return 300;
  }

  @Override
  public double numberOfSymbols() {
    return 2.;
  }

//  @Override
//  public void verify(final Tester trainer) {
//    trainer.verifyConvergence(0.05, 1);
//  }

  @Override
  public int width() {
    return 300;
  }

}
