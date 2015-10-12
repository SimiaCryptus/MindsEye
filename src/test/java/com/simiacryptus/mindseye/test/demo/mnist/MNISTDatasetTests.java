package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;

public class MNISTDatasetTests {

  protected static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  protected int getSampleSize(final Integer populationIndex, final int defaultNum) {
    return defaultNum;
  }

  public boolean filter(final LabeledObject<NDArray> item) {
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
    final NDArray[] trainingData = transformDataSet(MNIST.trainingDataStream(), limit, hash);
    final NDArray[] validationData = transformDataSet(MNIST.validationDataStream(), limit, hash);
    final Map<BufferedImage, String> report = new java.util.LinkedHashMap<>();
    try {
      evaluateImageList(trainingData).stream().forEach(i->report.put(i, "TRAINING"));
      evaluateImageList(validationData).stream().forEach(i->report.put(i, "VALIDATION"));
    } finally {
      final Stream<String> map = report.entrySet().stream().map(e -> Util.toInlineImage(e.getKey(), e.getValue().toString()));
      Util.report(map.toArray(i -> new String[i]));
    }
  }

  public List<BufferedImage> evaluateImageList(final NDArray[] validationData) {
    return java.util.Arrays.stream(validationData).map(Util::toImage).collect(java.util.stream.Collectors.toList());
  }

  public NDArray[] transformDataSet(Stream<LabeledObject<NDArray>> trainingDataStream, int limit, final int hash) {
    return trainingDataStream
        .collect(java.util.stream.Collectors.toList()).stream().parallel()
        .filter(this::filter)
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        .limit(limit)
        .map(obj -> obj.data)
        .toArray(i1 -> new NDArray[i1]);
  }

}
