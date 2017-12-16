# ConvolutionLayer
## IrregularTest_Float
### Json Serialization
Code from [JsonTest.java:36](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
      "id": "f4eb68e3-0cd0-493c-bf8d-ab007cc1e713",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4eb68e3-0cd0-493c-bf8d-ab007cc1e713",
      "filter": [
        [
          [
            -1.184,
            -0.76,
            -1.88
          ],
          [
            -0.7,
            0.42,
            -1.236
          ],
          [
            0.56,
            0.108,
            1.732
          ]
        ],
        [
          [
            1.7,
            0.512,
            -1.24
          ],
          [
            -0.424,
            1.992,
            0.0
          ],
          [
            1.036,
            -0.256,
            0.0
          ]
        ],
        [
          [
            1.204,
            1.772,
            -1.396
          ],
          [
            -0.032,
            -1.008,
            0.56
          ],
          [
            0.556,
            -0.936,
            0.52
          ]
        ],
        [
          [
            1.604,
            1.252,
            1.576
          ],
          [
            1.616,
            0.296,
            0.22
          ],
          [
            1.084,
            -1.572,
            1.584
          ]
        ],
        [
          [
            0.0
```
...[skipping 5119 bytes](etc/105.txt)...
```
     -0.484,
            0.392,
            1.804
          ],
          [
            -1.36,
            0.724,
            0.248
          ],
          [
            0.988,
            0.496,
            -0.48
          ]
        ],
        [
          [
            -0.224,
            0.08,
            -0.448
          ],
          [
            1.204,
            1.684,
            0.492
          ],
          [
            1.872,
            0.312,
            1.396
          ]
        ],
        [
          [
            -0.3,
            -0.344,
            -1.72
          ],
          [
            -0.424,
            -1.564,
            -0.596
          ],
          [
            1.572,
            0.616,
            -1.484
          ]
        ],
        [
          [
            0.956,
            -1.62,
            0.64
          ],
          [
            0.912,
            -0.532,
            1.616
          ],
          [
            1.824,
            -0.304,
            -1.844
          ]
        ],
        [
          [
            -1.128,
            1.308,
            0.908
          ],
          [
            -0.788,
            1.08,
            1.688
          ],
          [
            0.648,
            -1.232,
            -1.72
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "precision": "Float"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.02 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -0.556, 0.724, -1.716, -1.992, 1.98, 0.456, 1.22 ], [ 1.816, -1.9, 1.8, -0.104, 1.52, -1.052, 0.564 ], [ -1.768, 1.092, -0.672, -1.612, 1.612, 1.556, 1.412 ], [ -1.64, -0.916, -1.328, -1.1, 0.796, 1.768, -1.324 ], [ 0.764, 1.656, 0.448, 0.368, -0.632, -0.456, -0.296 ] ],
    	[ [ 0.984, 0.592, 1.86, -1.952, 1.16, -1.336, 0.764 ], [ 0.736, 1.676, -1.9, 0.088, 1.852, 1.676, 0.908 ], [ 1.144, 0.444, 0.924, -1.708, -0.268, -1.84, 1.912 ], [ 1.612, 1.988, 1.276, -0.104, -1.036, 0.044, 0.648 ], [ 0.124, -1.884, 1.52, 0.904, 0.572, 1.384, -1.012 ] ],
    	[ [ -0.92, 0.956, -1.652, -0.668, 1.672, 1.66, 0.136 ], [ -1.916, -1.104, 1.368, 1.984, 0.16, -1.204, -0.184 ], [ -1.932, -0.164, 1.552, -0.664, 0.444, -1.776, -1.42 ], [ -1.508, 1.996, -1.232, 0.4, -1.536, -0.476, -0.172 ], [ 0.196, -1.464, -0.52, -0.192, 1.348, 0.556, 1.916 ] ],
    	[ [ -1.236, -0.988, 1.244, -1.592, 0.444, -1.832, 1.16 ], [ 0.92, -0.048, 0.964, -1.512, -0.252, -0.592, -1.576 ], [ 1.604, -0.08, 0.672, -1.612, 0.588, 0.98
```
...[skipping 1200 bytes](etc/106.txt)...
```
    --------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3000#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.192, 0.132, 0.044, 0.376, 1.512, 0.548, -0.412 ], [ -0.208, 1.888, -0.412, 1.724, -0.5, 0.852, 0.792 ], [ -0.808, -1.952, 0.196, -0.212, -0.756, 1.676, 1.232 ], [ -1.312, 0.768, -0.48, 0.288, 0.352, -0.76, -0.656 ], [ -0.76, -0.56, 0.76, 0.464, 1.132, 0.932, 0.328 ] ],
    	[ [ -0.18, -0.26, 0.096, 1.04, -1.8, 0.548, -1.264 ], [ -0.056, -1.468, 1.388, 1.66, -0.412, 1.728, -0.48 ], [ 1.652, -0.744, -1.512, -1.4, 0.98, 0.284, 1.096 ], [ 1.692, -0.604, -1.76, -1.168, 1.756, 0.048, -0.544 ], [ 1.62, -1.4, 1.392, -1.688, -1.948, 1.576, -1.568 ] ],
    	[ [ 1.896, -1.112, 1.996, -0.116, 1.788, -1.144, -0.264 ], [ -1.66, -0.956, -1.424, -1.808, -0.336, -1.408, 0.8 ], [ 0.572, -1.344, -0.14, -1.64, -0.1, -1.92, 0.072 ], [ -0.392, -1.176, 0.572, -0.208, -1.732, 1.736, 0.156 ], [ -0.788, 0.884, 0.752, -1.128, 0.372, -1.428, 0.488 ] ],
    	[ [ 0.852, 1.056, 1.736, -0.736, -1.492, -1.672, 1.816 ], [ 1.132, 0.188, 1.42, -0.876, 1.188, 1.132, 1.804 ], [ 1.904, 1.52, 0.372, 0.228, 0.428, -1.288, 1.468 ], [ -0.144, -1
```
...[skipping 7073 bytes](etc/107.txt)...
```
    20800000429153442, 0.0560000017285347, 1.659999966621399, -1.1319999694824219, 0.0, 0.8080000281333923, -1.6519999504089355, ... ], [ 0.18000000715255737, -1.8960000276565552, -0.8519999980926514, -0.7039999961853027, 0.0, 0.0560000017285347, 1.659999966621399, -1.1319999694824219, ... ], [ 0.19200000166893005, 0.18000000715255737, -1.8960000276565552, -0.8519999980926514, -0.7039999961853027, 0.20800000429153442, 0.0560000017285347, 1.659999966621399, ... ], [ 0.0, 0.19200000166893005, 0.18000000715255737, -1.8960000276565552, -0.8519999980926514, 0.0, 0.20800000429153442, 0.0560000017285347, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.18000000715255737, -1.8960000276565552, -0.8519999980926514, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.19200000166893005, 0.18000000715255737, -1.8960000276565552, ... ], ... ]
    Error Statistics: {meanExponent=-0.15559017827335672, negative=2865, min=0.656000018119812, max=0.656000018119812, mean=-0.0036226031595752353, count=39375.0, positive=3050, stdDev=0.4446402642675351, zeros=33460}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.4643e-01 +- 4.1985e-01 [0.0000e+00 - 1.9960e+00] (39375#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (5915#)}
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.lambda$test$11(SingleDerivativeTester.java:150)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.test(SingleDerivativeTester.java:183)
    	at com
```
...[skipping 2957 bytes](etc/108.txt)...
```
    unner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runners.Suite.runChild(Suite.java:128)
    	at org.junit.runners.Suite.runChild(Suite.java:27)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runner.JUnitCore.run(JUnitCore.java:137)
    	at com.intellij.junit4.JUnit4IdeaTestRunner.startRunnerWithArgs(JUnit4IdeaTestRunner.java:68)
    	at com.intellij.rt.execution.junit.IdeaTestRunner$Repeater.startRunnerWithArgs(IdeaTestRunner.java:47)
    	at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    
```



