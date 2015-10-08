package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.Graphics2D;
import java.io.IOException;
import java.util.Iterator;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.LabeledObject;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.test.Tester;

public class SimpliedConvolutionLearningTest2 extends SimpliedConvolutionLearningTest {

  public Stream<LabeledObject<NDArray>> getTrainingData() throws IOException {
    
    final Stream<LabeledObject<NDArray>> merged = Util.toStream(new Iterator<LabeledObject<NDArray>>() {
      @Override
      public boolean hasNext() {
        return true;
      }

      int cnt=0;
      @Override
      public LabeledObject<NDArray> next() {
        int index = cnt++;
        String id;
        NDArray imgData;
        while(true){
          java.awt.image.BufferedImage img = new java.awt.image.BufferedImage(28,28,java.awt.image.BufferedImage.TYPE_BYTE_GRAY);
          Graphics2D g = img.createGraphics();
          int cardinality = index%4;
          if(0==cardinality)
          {
            int x1 = Util.R.get().nextInt(28);
            int x2 = Util.R.get().nextInt(28);
            int y = Util.R.get().nextInt(28);
            if(Math.abs(x1-x2)<1) continue;
            g.drawLine(x1, y, x2, y);
            id = "[0]";
          } else if(1==cardinality) {
            int x = Util.R.get().nextInt(28);
            int y1 = Util.R.get().nextInt(28);
            int y2 = Util.R.get().nextInt(28);
            if(Math.abs(y1-y2)<1) continue;
            g.drawLine(x, y1, x, y2);
            id = "[1]";
          } else if(2==cardinality) {
            int x1 = Util.R.get().nextInt(28);
            int x2 = Util.R.get().nextInt(28);
            int y = Util.R.get().nextInt(28);
            if(Math.abs(x1-x2)<1) continue;
            g.drawLine(x1, y, x2, y);
            int x = Util.R.get().nextInt(28);
            int y1 = Util.R.get().nextInt(28);
            int y2 = Util.R.get().nextInt(28);
            if(Math.abs(y1-y2)<1) continue;
            g.drawLine(x, y1, x, y2);
            id = "[2]";
          } else {
            int x = Util.R.get().nextInt(28);
            int y = Util.R.get().nextInt(28);
            int l = Util.R.get().nextInt(5);
            if(x+l<0||x+l>=28) continue;
            if(y+l<0||y+l>=28) continue;
            g.drawLine(x, y, x+l, y+l);
            id = "[3]";
          }
          imgData = new NDArray(new int[]{28,28,1}, img.getData().getSamples(0, 0, 28, 28, 0, (double[])null));
          break;
        }
        return new LabeledObject<NDArray>(imgData, id);
      }
    }, 1000).limit(1000);
    return merged;
  }

  @Override
  public void verify(final Tester trainer) {
    trainer.verifyConvergence(.5, 1,0);
  }

  public double numberOfSymbols() {
    return 4.;
  }

}
