# SimpleConvolutionLayer
## Image_Float
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "6b2a05f7-3a89-47b7-97ad-55b78c3c7ac0",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/6b2a05f7-3a89-47b7-97ad-55b78c3c7ac0",
      "filter": [
        [
          [
            1.096,
            1.12,
            -1.828
          ],
          [
            0.24,
            0.572,
            -1.96
          ],
          [
            1.332,
            0.956,
            -0.056
          ]
        ],
        [
          [
            -0.644,
            -1.732,
            0.752
          ],
          [
            1.468,
            -0.82,
            0.428
          ],
          [
            1.092,
            0.244,
            1.008
          ]
        ],
        [
          [
            1.668,
            1.448,
            0.32
          ],
          [
            0.888,
            -1.42,
            1.656
          ],
          [
            1.492,
            0.636,
            -1.284
          ]
        ],
        [
          [
            0.464,
            -0.728,
            0.364
          ],
          [
            -1.208,
            1.096,
            1.172
          ],
          [
            1.22,
            1.9,
            0.932
          ]
        ],
        [
       
```
...[skipping 28 bytes](etc/193.txt)...
```
    -1.392,
            0.036
          ],
          [
            -1.072,
            -1.188,
            1.356
          ],
          [
            1.628,
            1.404,
            -0.716
          ]
        ],
        [
          [
            -1.14,
            -1.86,
            -1.596
          ],
          [
            -1.936,
            -0.324,
            0.716
          ],
          [
            1.292,
            0.552,
            -1.604
          ]
        ],
        [
          [
            1.108,
            0.632,
            1.964
          ],
          [
            0.272,
            0.236,
            -0.412
          ],
          [
            -1.492,
            1.78,
            1.112
          ]
        ],
        [
          [
            1.2,
            0.912,
            1.748
          ],
          [
            -1.576,
            0.92,
            -1.632
          ],
          [
            1.66,
            0.98,
            1.996
          ]
        ],
        [
          [
            -0.436,
            -1.116,
            0.396
          ],
          [
            0.912,
            -0.28,
            1.496
          ],
          [
            -1.712,
            1.956,
            1.0
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false,
      "precision": "Float"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    	[ [ -0.592, -0.12, -0.564 ], [ 1.04, 1.956, -1.164 ], [ -1.608, 1.708, 0.212 ] ],
    	[ [ 0.688, -1.172, -0.712 ], [ -1.176, -0.772, -0.948 ], [ -0.476, -0.508, 0.38 ] ],
    	[ [ 1.984, -1.572, 0.516 ], [ 1.432, -0.032, -0.296 ], [ 1.904, 1.072, 1.684 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
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
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (540#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.504, -0.292, -1.0 ], [ -1.092, -1.728, 1.744 ], [ -1.148, 0.78, -1.66 ] ],
    	[ [ -1.548, -0.48, 0.5 ], [ 0.604, 0.584, -1.008 ], [ 0.528, 0.696, 0.888 ] ],
    	[ [ -0.544, 1.832, 0.268 ], [ 1.164, -0.028, -0.62 ], [ -1.468, -0.78, -1.12 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13585111063803984, negative=15, min=-1.12, max=-1.12, mean=-0.16385185185185183, count=27.0, positive=12, stdDev=1.0210481072463655, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=27.0, positive=0, stdDev=0.0, zeros=27}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.504, -0.292, -1.0 ], [ -1.092, -1.728, 1.744 ], [ -1.148, 0.78, -1.66 ] ],
    	[ [ -1.548, -0.48, 0.5 ], [ 0.604, 0.584, -1.008 ], [ 0.528, 0.696, 0.888 ] ],
    	[ [ -0.544, 1.832, 0.268 ], [ 1.164, -0.028, -0.62 ], [ -1.468
```
...[skipping 3967 bytes](etc/194.txt)...
```
    36, 0.0, 0.0, ... ], [ 0.0, 1.0920000076293945, -0.6039999723434448, 0.0, 1.1480000019073486, -0.527999997138977, 0.0, 0.0, ... ], [ 1.5479999780654907, 0.5440000295639038, 0.0, -0.6039999723434448, -1.1640000343322754, 0.0, -0.527999997138977, 1.468000054359436, ... ], [ -0.5040000081062317, 1.5479999780654907, 0.5440000295639038, 1.0920000076293945, -0.6039999723434448, -1.1640000343322754, 1.1480000019073486, -0.527999997138977, ... ], [ 0.0, -0.5040000081062317, 1.5479999780654907, 0.0, 1.0920000076293945, -0.6039999723434448, 0.0, 1.1480000019073486, ... ], [ 0.0, 0.0, 0.0, 1.5479999780654907, 0.5440000295639038, 0.0, -0.6039999723434448, -1.1640000343322754, ... ], [ 0.0, 0.0, 0.0, -0.5040000081062317, 1.5479999780654907, 0.5440000295639038, 1.0920000076293945, -0.6039999723434448, ... ], ... ]
    Error Statistics: {meanExponent=-0.14356775949935804, negative=210, min=1.0080000162124634, max=1.0080000162124634, mean=0.02297393802875354, count=2187.0, positive=231, stdDev=0.45435810224420303, zeros=1746}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.8031e-01 +- 4.1768e-01 [0.0000e+00 - 1.8320e+00] (2187#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (441#)}
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
    	at com.s
```
...[skipping 2955 bytes](etc/195.txt)...
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



