package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.test.MNIST;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;

public class MNISTDatasetTests {

  protected static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

    public static void report(final NNLayer<DAGNetwork> net) throws FileNotFoundException, IOException {
      final File outDir = new File("reports");
      outDir.mkdirs();
      final StackTraceElement caller = Thread.currentThread().getStackTrace()[2];
      final File report = new File(outDir, caller.getClassName() + "_" + caller.getLineNumber() + ".html");
      final PrintStream out = new PrintStream(new FileOutputStream(report));
      out.println("<html><head></head><body>");
      MNIST.trainingDataStream().sorted(Comparator.comparing(img -> img.label))
          .map(x -> "<p>" + Util.toInlineImage(x.<BufferedImage>map(tensor -> tensor.toRgbImage())) + net.eval(x.data).data.toString() + "</p>").forEach(out::println);
      out.println("</body></html>");
      out.close();
      Desktop.getDesktop().browse(report.toURI());
    }

    protected int getSampleSize(final Integer populationIndex, final int defaultNum) {
    return defaultNum;
  }

  public boolean filter(final LabeledObject<Tensor> item) {
    if (item.label.equals("[0]"))
      return true;
    if (item.label.equals("[5]"))
      return true;
    if (item.label.equals("[9]"))
      return true;
    return true;
  }

  @Test
  public void test() throws Exception {
    final int hash = Util.R.get().nextInt();
    log.debug(String.format("Shuffle hash: 0x%s", Integer.toHexString(hash)));
    int limit = 1000;
    final Tensor[] trainingData = transformDataSet(MNIST.trainingDataStream(), limit, hash);
    final Tensor[] validationData = transformDataSet(MNIST.validationDataStream(), limit, hash);
    final Map<BufferedImage, String> report = new java.util.LinkedHashMap<>();
    try {
      evaluateImageList(trainingData).stream().forEach(i->report.put(i, "TRAINING"));
      evaluateImageList(validationData).stream().forEach(i->report.put(i, "VALIDATION"));
    } finally {
      final Stream<String> map = report.entrySet().stream().map(e -> Util.toInlineImage(e.getKey(), e.getValue().toString()));
      Util.report(map.toArray(i -> new String[i]));
    }
  }

  public List<BufferedImage> evaluateImageList(final Tensor[] validationData) {
    return java.util.Arrays.stream(validationData).map(tensor -> tensor.toRgbImage()).collect(java.util.stream.Collectors.toList());
  }

  public Tensor[] transformDataSet(Stream<LabeledObject<Tensor>> trainingDataStream, int limit, final int hash) {
    return trainingDataStream
        .collect(java.util.stream.Collectors.toList()).stream().parallel()
        .filter(this::filter)
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        .limit(limit)
        .map(obj -> obj.data)
        .toArray(i1 -> new Tensor[i1]);
  }

}
