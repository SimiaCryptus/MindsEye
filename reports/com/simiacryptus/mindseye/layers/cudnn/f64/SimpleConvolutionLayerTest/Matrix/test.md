# SimpleConvolutionLayer
## Matrix
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b64",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-504000002b64",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -0.1,
          1.168,
          0.384,
          1.608,
          -1.576,
          -1.448,
          -0.284,
          -1.604,
          1.508
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.228 ], [ 0.044 ], [ 1.076 ] ],
    	[ [ 0.796 ], [ 1.116 ], [ -1.304 ] ],
    	[ [ -1.184 ], [ -1.896 ], [ -1.348 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.8604320000000002 ], [ 2.520576 ], [ -4.180128000000001 ] ],
    	[ [ -1.978528 ], [ -6.443088000000001 ], [ -2.855776 ] ],
    	[ [ -1.0726079999999996 ], [ 2.396432 ], [ 8.736752000000001 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b65",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000002b65",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -0.1,
          1.168,
          0.384,
          1.608,
          -1.576,
          -1.448,
          -0.284,
          -1.604,
          1.508
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 0.228 ], [ 0.044 ], [ 1.076 ] ],
    	[ [ 0.796 ], [ 1.116 ], [ -1.304 ] ],
    	[ [ -1.184 ], [ -1.896 ], [ -1.348 ] ]
    ]
    Error: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.9343e-18 +- 4.6550e-17 [0.0000e+00 - 4.4409e-16] (180#), relativeTol=7.1720e-18 +- 6.7660e-17 [0.0000e+00 - 6.4548e-16] (180#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.228 ], [ 0.044 ], [ 1.076 ] ],
    	[ [ 0.796 ], [ 1.116 ], [ -1.304 ] ],
    	[ [ -1.184 ], [ -1.896 ], [ -1.348 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.15800726931773432, negative=4, min=-1.348, max=-1.348, mean=-0.2746666666666666, count=9.0, positive=5, stdDev=1.1018158749184104, zeros=0}
    Output: [
    	[ [ 0.8604320000000002 ], [ 2.520576 ], [ -4.180128000000001 ] ],
    	[ [ -1.978528 ], [ -6.443088000000001 ], [ -2.855776 ] ],
    	[ [ -1.0726079999999996 ], [ 2.396432 ], [ 8.736752000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=0.4299913871388419, negative=5, min=8.736752000000001, max=8.736752000000001, mean=-0.22399288888888888, count=9.0, positive=4, stdDev=4.2285167311800285, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.228 ], [ 0.044 ], [ 1.076 ] ],
    	[ [ 0.796 ], [ 1.116 ], [ -1.304 ] ],
    	[ [ -1.184 ], [ -1.896 ], [ -1.348 ] ]
    ]
    Value Statistics: {meanExponent=-0.15800726931773432, negative=4, min=-1.348, max=-1.348, mean=-0.2746666666666666, count=9.0, positive=5, stdDev=1.1018158749184104, zeros=0}
    Implemented Feedback: [ [ -1.576, -1.448, 0.0, -1.604, 1.508, 0.0, 0.0, 0.0, 0.0 ], [ 1.608, -1.576, -1.448, -0.284, -1.604, 1.508, 0.0, 0.0, 0.0 ], [ 0.0, 1.608, -1.576, 0.0, -0.284, -1.604, 0.0, 0.0, 0.0 ], [ 1.168, 0.384, 0.0, -1.576, -1.448, 0.0, -1.604, 1.508, 0.0 ], [ -0.1, 1.168, 0.384, 1.608, -1.576, -1.448, -0.284, -1.604, 1.508 ], [ 0.0, -0.1, 1.168, 0.0, 1.608, -1.576, 0.0, -0.284, -1.604 ], [ 0.0, 0.0, 0.0, 1.168, 0.384, 0.0, -1.576, -1.448, 0.0 ], [ 0.0, 0.0, 0.0, -0.1, 1.168, 0.384, 1.608, -1.576, -1.448 ], [ 0.0, 0.0, 0.0, 0.0, -0.1, 1.168, 0.0, 1.608, -1.576 ] ]
    Implemented Statistics: {meanExponent=-0.031011828159925734, negative=29, min=-1.576, max=-1.576, mean=-0.12108641975308641, count=81.0, positive=20, stdDev=1.0118527218934736, zeros=32}
    Measured Feedback: [ [ -1.5759999999997998, -1.4479999999994497, 0.0, -1.6040000000039356, 1.508000000001175, 0.0, 0.0, 0.0, 0.0 ], [ 1.6079999999996097, -1.57600000000091, -1.4479999999994497, -0.28400000000150527, -1.604000000
```
...[skipping 4857 bytes](etc/1.txt)...
```
    99999990448, 1.1160000000032255 ] ]
    Measured Statistics: {meanExponent=-0.15459137520901314, negative=20, min=1.1160000000032255, max=1.1160000000032255, mean=-0.11145679012315295, count=81.0, positive=29, stdDev=0.8854269814565351, zeros=32}
    Gradient Error: [ [ -1.2154721673596214E-12, 9.903189379656396E-13, 0.0, 2.4706903190008234E-12, 3.425926209388308E-12, 0.0, 0.0, 0.0, 0.0 ], [ -9.551942570240612E-13, 1.0049738818906917E-12, -1.2301271112846734E-12, 2.0752288776293426E-12, 2.4706903190008234E-12, 3.425926209388308E-12, 0.0, 0.0, 0.0 ], [ 0.0, -9.551942570240612E-13, 1.0049738818906917E-12, 0.0, 6.516120976129969E-12, 2.4706903190008234E-12, 0.0, 0.0, 0.0 ], [ -1.5354384430565915E-12, -7.400746682151293E-13, 0.0, 3.2254199311410048E-12, 3.2107649872159527E-12, 0.0, 2.4706903190008234E-12, -1.0149658891123181E-12, 0.0 ], [ -1.0505485370515544E-13, -1.5354384430565915E-12, -7.400746682151293E-13, -9.551942570240612E-13, 3.2254199311410048E-12, -1.2301271112846734E-12, 6.516120976129969E-12, -1.9702017794998028E-12, 3.425926209388308E-12 ], [ 0.0, -1.0505485370515544E-13, -1.5354384430565915E-12, 0.0, -9.551942570240612E-13, -1.2154721673596214E-12, 0.0, -2.3656632208712836E-12, 2.4706903190008234E-12 ], [ 0.0, 0.0, 0.0, -1.5354384430565915E-12, 1.4803713810351837E-12, 0.0, 3.2254199311410048E-12, -1.2301271112846734E-12, 0.0 ], [ 0.0, 0.0, 0.0, -1.0505485370515544E-13, 2.9054536554440347E-12, -2.9605207174654424E-12, -9.551942570240612E-13, -1.2154721673596214E-12, -5.6710192097853E-12 ], [ 0.0, 0.0, 0.0, 0.0, -1.0505485370515544E-13, -1.5354384430565915E-12, 0.0, -9.551942570240612E-13, 3.2254199311410048E-12 ] ]
    Error Statistics: {meanExponent=-11.841876533435169, negative=28, min=3.2254199311410048E-12, max=3.2254199311410048E-12, mean=3.038190412557938E-13, count=81.0, positive=21, stdDev=1.8981635438160423E-12, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2659e-12 +- 1.6901e-12 [0.0000e+00 - 1.0552e-11] (162#)
    relativeTol: 2.0809e-12 +- 3.5741e-12 [6.3542e-14 - 1.8929e-11] (98#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2659e-12 +- 1.6901e-12 [0.0000e+00 - 1.0552e-11] (162#), relativeTol=2.0809e-12 +- 3.5741e-12 [6.3542e-14 - 1.8929e-11] (98#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.16 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.7617 +- 0.9059 [5.0755 - 12.3424]
    Learning performance: 3.7411 +- 0.3643 [3.3172 - 4.7620]
    
```

