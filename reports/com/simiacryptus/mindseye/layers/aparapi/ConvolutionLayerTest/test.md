# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.04 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          -1.54,
          0.832,
          -0.156,
          -1.488,
          -0.088,
          0.128,
          0.044,
          -0.444,
          -0.076,
          0.324,
          0.728,
          1.936,
          1.636,
          1.0,
          -1.624,
          -0.004,
          0.072,
          -0.628,
          -1.484,
          0.932,
          -1.004,
          -0.76,
          -1.18,
          1.4,
          -1.372,
          1.06,
          1.772,
          -0.2,
          -0.688,
          -1.256,
          1.816,
          1.268,
          1.6,
          -0.544,
          1.656,
          -0.264
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
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.31 seconds: 
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
    	[ [ 0.904, -0.456 ], [ -1.372, -1.828 ], [ -0.476, 0.096 ] ],
    	[ [ 1.544, -1.324 ], [ -1.384, 1.304 ], [ 1.764, 1.328 ] ],
    	[ [ 1.364, -1.272 ], [ -1.112, -1.68 ], [ -1.78, -0.948 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -3.4816800000000008, -0.0029760000000001734 ], [ -0.6480159999999999, -3.6686719999999995 ], [ -6.884, 1.113488000000001 ] ],
    	[ [ 6.159632000000001, -4.700192000000001 ], [ 4.4504480000000015, -8.198208 ], [ 2.8381439999999993, 4.063232 ] ],
    	[ [ -3.859296, -8.845696 ], [ -4.661696000000001, -0.4330239999999996 ], [ 4.489072, -6.059247999999999 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.11 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.55 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.904, -0.456 ], [ -1.372, -1.828 ], [ -0.476, 0.096 ] ],
    	[ [ 1.544, -1.324 ], [ -1.384, 1.304 ], [ 1.764, 1.328 ] ],
    	[ [ 1.364, -1.272 ], [ -1.112, -1.68 ], [ -1.78, -0.948 ] ]
    ]
    Inputs Statistics: {meanExponent=0.01938507057599784, negative=11, min=-0.948, max=-0.948, mean=-0.296, count=18.0, positive=7, stdDev=1.2723647275840368, zeros=0}
    Output: [
    	[ [ -3.4816800000000008, -0.0029760000000001734 ], [ -0.6480159999999999, -3.6686719999999995 ], [ -6.884, 1.113488000000001 ] ],
    	[ [ 6.159632000000001, -4.700192000000001 ], [ 4.4504480000000015, -8.198208 ], [ 2.8381439999999993, 4.063232 ] ],
    	[ [ -3.859296, -8.845696 ], [ -4.661696000000001, -0.4330239999999996 ], [ 4.489072, -6.059247999999999 ] ]
    ]
    Outputs Statistics: {meanExponent=0.36858962151195995, negative=12, min=-6.059247999999999, max=-6.059247999999999, mean=-1.573816, count=18.0, positive=6, stdDev=4.555746784093934, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.904, -0.456 ], [ -1.372, -1.828 ], [ -0.476, 0.096 ] ],
    	[ [ 1.544, -1.324 ], [ -1.384, 1.304 ], [ 1.764, 1.328 ] ],
    	[ [ 1.364, -1.272 ], [ -1.112, -1.68 ], [ -1.78, -0.948 ] ]
    ]
    Value Statistics: {meanExponent=0.01938507057599784, negative=11, min=-0.948, max=-0.948, mean=-0.296, count=18.0, positive=7, stdDev=1.2723647275840368, zeros=0}
    Implemented Feedback: [ [ -0.088, 0.128, 0.0, -0.444, -0.076, 0.0, 0.0, 0.0, ... ], [ -1.488, -0.088, 0.128, 0.044, -0.444, -0.076, 0.0, 0.0, ... ], [ 0.0, -1.488, -0.088, 0.0, 0.044, -0.444, 0.0, 0.0, ... ], [ 0.832, -0.156, 0.0, -0.088, 0.128, 0.0, -0.444, -0.076, ... ], [ -1.54, 0.832, -0.156, -1.488, -0.088, 0.128, 0.044, -0.444, ... ], [ 0.0, -1.54, 0.832, 0.0, -1.488, -0.088, 0.0, 0.044, ... ], [ 0.0, 0.0, 0.0, 0.832, -0.156, 0.0, -0.088, 0.128, ... ], [ 0.0, 0.0, 0.0, -1.54, 0.832, -0.156, -1.488, -0.088, ... ], ... ]
    Implemented Statistics: {meanExponent=-0.22374532774918393, negative=96, min=1.268, max=1.268, mean=0.09977777777777774, count=324.0, positive=100, stdDev=0.8530040163163662, zeros=128}
    Measured Feed
```
...[skipping 4407 bytes](etc/1.txt)...
```
    5440000000044307, 1.364000000005916, 0.0, -1.3839999999998298, -1.11200000000089, ... ], [ 0.0, 0.0, 0.0, 0.9040000000037907, 1.543999999995549, 1.3639999999970343, -1.3720000000017052, -1.3839999999998298, ... ], ... ]
    Measured Statistics: {meanExponent=0.054895368432822875, negative=118, min=1.3039999999975294, max=1.3039999999975294, mean=-0.08355555555545623, count=648.0, positive=78, stdDev=0.7355135268287774, zeros=452}
    Gradient Error: [ [ 1.7008616737257398E-13, -8.899547765395255E-13, 0.0, -3.452793606584237E-13, -1.1479706074624119E-13, 0.0, 0.0, 0.0, ... ], [ 7.176703675781937E-12, 1.7008616737257398E-13, -8.899547765395255E-13, 6.296296817254188E-12, 8.536504836342829E-12, -1.1479706074624119E-13, 0.0, 0.0, ... ], [ 0.0, -1.7050805212193154E-12, 1.7008616737257398E-13, 0.0, -2.5854873797470646E-12, -3.452793606584237E-13, 0.0, 0.0, ... ], [ 4.430678046674075E-12, -2.965849787983643E-12, 0.0, 1.7008616737257398E-13, 7.991829420461727E-12, 0.0, -3.452793606584237E-13, -1.1479706074624119E-13, ... ], [ 3.790634472977672E-12, -4.451106150327178E-12, -2.965849787983643E-12, -1.7050805212193154E-12, 9.051870364373826E-12, -8.899547765395255E-13, -2.5854873797470646E-12, -3.452793606584237E-13, ... ], [ 0.0, -5.09114972402358E-12, -1.021405182655144E-14, 0.0, 7.176703675781937E-12, 1.7008616737257398E-13, 0.0, 6.296296817254188E-12, ... ], [ 0.0, 0.0, 0.0, 4.430678046674075E-12, 5.915934409017609E-12, 0.0, 1.7008616737257398E-13, -8.899547765395255E-13, ... ], [ 0.0, 0.0, 0.0, 3.790634472977672E-12, -4.451106150327178E-12, -2.965849787983643E-12, -1.7050805212193154E-12, 1.7008616737257398E-13, ... ], ... ]
    Error Statistics: {meanExponent=-11.852734661226258, negative=108, min=-2.4706903190008234E-12, max=-2.4706903190008234E-12, mean=9.933900410082949E-14, count=648.0, positive=88, stdDev=2.1775905327841166E-12, zeros=452}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2375e-12 +- 2.3270e-12 [0.0000e+00 - 1.5668e-11] (972#)
    relativeTol: 7.2277e-12 +- 4.8458e-11 [2.4328e-15 - 8.1829e-10] (392#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2375e-12 +- 2.3270e-12 [0.0000e+00 - 1.5668e-11] (972#), relativeTol=7.2277e-12 +- 4.8458e-11 [2.4328e-15 - 8.1829e-10] (392#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.78 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 26.9436 +- 2.9922 [23.1175 - 40.9971]
    Learning performance: 21.1785 +- 1.9249 [18.5635 - 30.7663]
    
```

