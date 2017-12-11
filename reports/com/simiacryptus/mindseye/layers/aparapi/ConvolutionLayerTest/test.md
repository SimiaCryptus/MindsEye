# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.04 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00000001",
      "filter": [
        [
          [
            -1.956,
            0.564,
            -1.004
          ],
          [
            0.436,
            0.64,
            -0.576
          ],
          [
            -0.696,
            -0.216,
            0.58
          ]
        ],
        [
          [
            -0.988,
            1.364,
            0.5
          ],
          [
            0.652,
            0.696,
            1.816
          ],
          [
            -1.448,
            -0.268,
            1.852
          ]
        ],
        [
          [
            0.112,
            -1.204,
            -1.156
          ],
          [
            0.896,
            -1.852,
            -0.168
          ],
          [
            -0.888,
            -0.084,
            -0.972
          ]
        ],
        [
          [
            1.316,
            -0.096,
            -0.188
          ],
          [
            -1.252,
            1.424,
            1.204
          ],
          [
            -1.752,
            -1.032,
            1.656
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": true
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.27 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -0.06, -0.696 ], [ -0.068, 1.368 ], [ -1.504, -0.18 ] ],
    	[ [ 1.56, -1.912 ], [ -0.176, -0.244 ], [ 0.072, 0.48 ] ],
    	[ [ -0.7, 0.336 ], [ 0.948, 0.084 ], [ -0.3, 0.42 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.1508960000000004, 2.006784 ], [ -2.9075040000000003, 2.443103999999999 ], [ 0.07121599999999997, -2.568336 ] ],
    	[ [ 1.5232639999999997, -4.794847999999999 ], [ 3.1059999999999994, 2.465984 ], [ -1.74584, -2.0172160000000003 ] ],
    	[ [ -0.7553120000000001, 1.7650560000000006 ], [ 2.177712, -0.7740319999999992 ], [ -1.168688, 0.027183999999999993 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:178](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L178) executed in 0.12 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.56 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.06, -0.696 ], [ -0.068, 1.368 ], [ -1.504, -0.18 ] ],
    	[ [ 1.56, -1.912 ], [ -0.176, -0.244 ], [ 0.072, 0.48 ] ],
    	[ [ -0.7, 0.336 ], [ 0.948, 0.084 ], [ -0.3, 0.42 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.44217444213225776, negative=10, min=0.42, max=0.42, mean=-0.031777777777777766, count=18.0, positive=8, stdDev=0.8442879679585078, zeros=0}
    Output: [
    	[ [ -1.1508960000000004, 2.006784 ], [ -2.9075040000000003, 2.443103999999999 ], [ 0.07121599999999997, -2.568336 ] ],
    	[ [ 1.5232639999999997, -4.794847999999999 ], [ 3.1059999999999994, 2.465984 ], [ -1.74584, -2.0172160000000003 ] ],
    	[ [ -0.7553120000000001, 1.7650560000000006 ], [ 2.177712, -0.7740319999999992 ], [ -1.168688, 0.027183999999999993 ] ]
    ]
    Outputs Statistics: {meanExponent=0.0902990641194955, negative=9, min=0.027183999999999993, max=0.027183999999999993, mean=-0.127576, count=18.0, positive=9, stdDev=2.168370052454249, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.06, -0.696 ], [ -0.068, 1.368 ], [ -1.504, -
```
...[skipping 6420 bytes](etc/1.txt)...
```
    . ], [ 4.951039578315886E-13, 4.4986236957811343E-13, -1.4499512701604544E-13, -4.17582635137137E-12, -6.201150704043812E-13, -1.6053824936079764E-12, 5.0182080713057076E-14, -1.500327639902821E-13, ... ], [ 0.0, 4.951039578315886E-13, 4.4986236957811343E-13, 0.0, 2.650657471292561E-13, -6.201150704043812E-13, 0.0, -1.0600409439120995E-12, ... ], [ 0.0, 0.0, 0.0, -3.991029728922513E-12, -1.4499512701604544E-13, 0.0, 4.901079542207754E-13, 6.150635556423367E-13, ... ], [ 0.0, 0.0, 0.0, 4.951039578315886E-13, 4.4986236957811343E-13, -1.4499512701604544E-13, -8.451572774959004E-13, -6.201150704043812E-13, ... ], ... ]
    Error Statistics: {meanExponent=-12.081672302672478, negative=95, min=2.65898414397725E-14, max=2.65898414397725E-14, mean=-5.956526424363325E-14, count=648.0, positive=101, stdDev=1.274723533738694E-12, zeros=452}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.4426e-13 +- 1.2759e-12 [0.0000e+00 - 8.5267e-12] (972#)
    relativeTol: 2.7874e-12 +- 5.6255e-12 [2.4328e-15 - 4.5217e-11] (392#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.4426e-13 +- 1.2759e-12 [0.0000e+00 - 8.5267e-12] (972#), relativeTol=2.7874e-12 +- 5.6255e-12 [2.4328e-15 - 4.5217e-11] (392#)}
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.80 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 26.7959 +- 3.2180 [21.9007 - 43.2086]
    Learning performance: 22.5817 +- 4.0813 [17.3752 - 42.8239]
    
```

