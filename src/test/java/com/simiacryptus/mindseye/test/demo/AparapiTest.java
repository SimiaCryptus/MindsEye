package com.simiacryptus.mindseye.test.demo;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.DAGNode;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.net.util.VerboseWrapper;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.test.LabeledObject;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public class AparapiTest {
  static final Logger log = LoggerFactory.getLogger(AparapiTest.class);

  public static final Random random = new Random();

  @Test
  public void test() throws Exception {
    float inA[] = new float[1024];
    float inB[] = new float[1024];
    assert (inA.length == inB.length);
    float[] result = new float[inA.length];

    Kernel kernel = new Kernel() {
      @Override
      public void run() {
        int i = getGlobalId();
        result[i] = inA[i] + inB[i];
      }
    };

    Range range = Range.create(result.length);
    kernel.execute(range);
  }


}
