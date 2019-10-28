PAC learning



boot

随机森林，并行训练



树模型，最好理解。








$$
= \sum_{i=1}^m [(\overline Y - w\overline X - b) - (wx_i - w\overline X - Y_i + \overline Y)]^2
$$
由于，
$$
= \sum_{i=1}^m [(\overline Y - w\overline X - b) - (wx_i - w\overline X - Y_i + \overline Y)]^2
\\ = \sum_{i=1}^m (\overline Y - w\overline X - b)^2 - \sum[(wx_i - w\overline X - Y_i + \overline Y)]^2
$$









$$
 
 \\ = \sum_{i=1}^{m}[(\overline Y - w\overline X - b)^2]  + \sum_{i=1}^{m}[(wx_i - w\overline X - Y_i + \overline Y)]^2
$$

















$$
\\ = m(\overline Y - w\overline X - b)^2 + \sum_{i=1}^{m}[(wx_i - w\overline X - Y_i + \overline Y)]^2  
 
 \\ = m(\overline Y - w\overline X - b)^2  + \sum_{i =1}^{m}[w(x_i - \overline X)- (Y_i - \overline Y)]^2
$$


