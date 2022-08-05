# =========================
#  Neural Network Factory
#   for PPSML class 2018
#   written by W.S.Cho
# =========================

import numpy as np

class MLP:

    model_nametag = "myMLP"
    
    model_structure, learning_rate = None, None
    n_layers, n_nodes, id_activations = None, None, None
    encoding = None
    
    # 가중치 & 편향
    W, B = None, None
    
    # 가중합(a_j=W_ji.f_i), 출력(f_j=f(a_j)), 오차보정델타(delta_j = dE/da_j)
    a, f, delta = None, None, None

    def __init__(self, model_nametag = 'a MLP', model_structure = '784:identity|100:tanh|10:softmax', learning_rate = 0.005, encoding = 'one-hot', load_model_np = None):
        """
        * MLP 초기화 입력변수 *
        1> model_nametag: 모델이름태그, (default: 'a MLP')
        2> model_structure: 모델구조, a string without blank, partitioned by ':' & '|'. 
            ex) 'input layer (784 nodes), 1-hidden (100 nodes, activation=tanh), output (10 nodes, activation=softmax)  
            => model_structure = '784:identity|100:tanh|10:softmax' (=default) 
            => 사용가능한 활성화 함수 : ["identity","sigmoid","tanh","relu","selu","softmax"]
        3> learning_rate: 학습률 , (default: 0.005) 
        4> encoding: 출력 정답 라벨의 표현 방식
            - 'one-hot': one-hot encoding
            - 'integer': integer value for each category
            - 'float': original float values of target regression function
        5> load_model_np: 신경망 모형 리로드하기 위해, 모형정보를 담고 있는 넘파이 배열을 입력 (default: None). 신경망으로 새로 생성하는 경우에는 무시.
            ex) load_model_np 구성예
            - type(load_model_np) = numpy.ndarray
            - load_model_np[0] = '*Neural Netowrk model in numpy ndarray*'
            - load_model_np[1] = model_nametag (a string)
            - load_model_np[2] = model_structure (a string)
            - load_model_np[3] = learning_rate
            - load_model_np[4] = target label encoding scheme
            - load_model_np[ 5:5+(self.n_layers-1) ] = weight parameters of neural net
            - load_model_np[ 5+(self.n_layers-1) : 5+2*(self.n_layers-1) ] : bias parameters of neural net
        """  
        
        # load_model_np의 유무에 따른 신경망 기본구성정보의 로드  
        if load_model_np is None:

            # 신경망 모형 태그정보 추출
            self.model_nametag = model_nametag
            # 신경망 구성 정보 추출
            self.model_structure = model_structure
            # 학습률
            self.learning_rate = learning_rate
            # 지도라벨값 encoding 방식
            self.encoding = encoding
            if self.encoding not in ['one-hot','integer','float']:
                raise ValueError('\n => 지도라벨의 인코딩 방식 변수값("encoding")을 확인하세요 ["one-hot", "integer", "float"]')

            layer_info_list = self.model_structure.split('|')
            self.n_layers = len(layer_info_list)
            self.n_nodes = [ int(layer_info_list[i].split(':')[0]) for i in range(self.n_layers) ]
            self.id_activations = [ layer_info_list[i].split(':')[1] for i in range(self.n_layers) ]

            activation_available = set(self.id_activations).issubset(['identity','relu','sigmoid','selu','softmax','tanh'])

            if not activation_available:
                raise ValueError('\n => 구성한 각층의 활성화 함수가 현재 지원하는 함수 목록에 있는지 확인하세요 \n 지원가능:("identity","sigmoid","tanh","relu","selu","softmax")')

            # 가중치(W, weight)와 편향치(B, bias) 파라메터의 초기화
            # W : 넘파이 배열(인접층들을 연결하는 가중치행렬)들의 리스트 => 리스트 요소수: n_layers-1 
            # B : 넘파이 배열(인접층들을 연결하는 편향치행렬)들의 리스트 => "
            # a[i+1] = W[i].f[i] + B[i], f[i] = f(a[i])   
            self.W, self.B = [], []
            for i in range(self.n_layers-1): # i = 0 .. n_layers-2
                if self.id_activations[i] in ['identity','selu','sigmoid','softmax','tanh']: # Xavier-init.
                    self.W.append(np.random.randn(self.n_nodes[i+1],self.n_nodes[i])/np.sqrt(self.n_nodes[i]))
                    self.B.append(np.random.randn(self.n_nodes[i+1],1)/np.sqrt(self.n_nodes[i]))

                elif self.id_activations[i] in ['relu']: # He-init.
                    self.W.append(np.random.randn(self.n_nodes[i+1],self.n_nodes[i])/np.sqrt(self.n_nodes[i]/2.))
                    self.B.append(np.random.randn(self.n_nodes[i+1],1)/np.sqrt(self.n_nodes[i]/2.))
                    
            # 각 노드에서의 가중합(a), 활성화출력(f), 오차항 델타(delta)의 초기화
            # a[i+1] = W[i].f[i] + B[i], f[i] = f(a[i])
            # a[0] = 입력층의 입력 (=x) 
            # f[0] = 입력층의 출력 (=x)
            # delta[n_layers-1] = 출력층의 오차*df_over_da; 이후 역전파에 사용
            self.a, self.f, self.delta = [None]*self.n_layers, [None]*self.n_layers, [None]*self.n_layers

            print ('\n * 다음과 같은 구조의 다층퍼셉트론 연결이 초기화 되었습니다 *\n')
            print (' > 모델이름 =',self.model_nametag)
            print (' > 총 층수 (입력 + 은닉(s) + 출력) = ', self.n_layers)
            print (' > 각 층에서의 노드수 = ', self.n_nodes)
            print (' > 각 층에서의 활성화 함수 = ', self.id_activations)
            print (' > 학습률(Learning Rate) = ', self.learning_rate)
            print (' > 지도라벨 인코딩 방식 = ', self.encoding)
        
        elif load_model_np is not None:
            
            if type(load_model_np) == np.ndarray:
            
                if load_model_np[0] == '*Neural Netowrk model in numpy ndarray*':
                    
                    # 신경망 모형 태그정보 추출
                    self.model_nametag = load_model_np[1]
                    # 신경망 구성 정보 추출
                    self.model_structure = load_model_np[2]
                    # 학습률
                    self.learning_rate = load_model_np[3]
                    # 지도라벨값 encoding 방식 
                    self.encoding = load_model_np[4]

                    layer_info_list = self.model_structure.split('|')
                    self.n_layers = len(layer_info_list)
                    self.n_nodes = [ int(layer_info_list[i].split(':')[0]) for i in range(self.n_layers) ]
                    self.id_activations = [ layer_info_list[i].split(':')[1] for i in range(self.n_layers) ]                
                   
                    # 모형의 로딩: 가중치(W, weight)와 편향치(B, bias) 파라메터 불러오기
                    self.W = load_model_np[ 5:5+(self.n_layers-1) ]
                    self.B = load_model_np[ 5+(self.n_layers-1) : 5+2*(self.n_layers-1) ]
                    
                    # 각 노드에서의 가중합(a), 활성화출력(f), 오차항 델타(delta)의 초기화
                    self.a, self.f, self.delta = [None]*self.n_layers, [None]*self.n_layers, [None]*self.n_layers

                    print ('\n * 다음과 같은 구조의 다층퍼셉트론 모형이 "load_model_np" 정보로부터 로드되었습니다. *\n')
                    print (' > 모델이름 =',self.model_nametag)
                    print (' > 총 층수 (입력 + 은닉(s) + 출력) = ', self.n_layers)
                    print (' > 각 층에서의 노드수 = ', self.n_nodes)
                    print (' > 각 층에서의 활성화 함수 = ', self.id_activations)
                    print (' > 학습률(Learning Rate) = ', self.learning_rate)
                    print (' > 지도라벨 인코딩 방식 = ', self.encoding)
                    
                else:
                    raise ValueError('=> Assign a valid model (in numpy.ndarray) to "load_model_np" (L2)')
            
            else:
                raise ValueError('=> Assign a valid model (in numpy.ndarray) to "load_model_np" (L1)')
                
       

    def feedforward(self, input_list):

        # input_x와 입력층의 size 확인
        if len(input_list) != self.W[0].shape[1]:
            raise ValueError(' => 입력층의 입력(input_x)변수의 차원과 입력층 노드수의 불일치!')
            
        # 데이터 입력(i=0층의 출력)으로부터 최종 출력층까지, 각 [i]층 노드에서의 가중합입력(a)과 출력(f)을 계산
        for i in range(self.n_layers):
            
            if i==0:
                self.a[i] = np.array(input_list, ndmin=2).T
                self.f[i] = self.a[i].copy()
                #self.f[i] = self.actfunc(self.a[i], func = self.id_activations[i])
            else:
                self.a[i] = np.matmul(self.W[i-1], self.f[i-1])  + self.B[i-1]
                self.f[i] = self.actfunc(self.a[i], func = self.id_activations[i])

        self.output_y = self.f[self.n_layers-1].copy()

        return self.output_y

 
    def backpropagate(self, target_list):
        
        # reshape target_y vector ( = a[i].shape = f[i].shape = output_y.shape)
        target_y = np.array(target_list, ndmin=2).T

        if target_y.shape != self.output_y.shape:
            print (' => output_y.shape = ', self.output_y.shape)
            print (' => target_y.shape = ', target_y.shape)            
            raise ValueError(' => 출력층의 출력과 지도라벨값의 numpy.array shape이 다름! ')
            
        # calculate dE_over_df at output layer (shape = shape of self.output_y, or target_y)
        # 1) for mean squared error (MSE) function:
        self.dE_over_df_at_output = (self.output_y - target_y)
        
        # calculate delta i = [n_layers-1 .. 1]
        for i in range(self.n_layers-1, 0, -1):
            
            if i == self.n_layers-1:              
                self.delta[i] = self.dE_over_df_at_output * self.df_over_da(self.a[i], func=self.id_activations[i])
            else:   
                self.delta[i] = np.matmul(self.W[i].T, self.delta[i+1])*self.df_over_da(self.a[i], func=self.id_activations[i])
                
        # update W(weights) & B(biases) based on the delta i = [n_layers-1 .. 1]
        for i in range(self.n_layers-1, 0, -1):
            self.W[i-1] += - self.learning_rate * np.matmul(self.delta[i], self.f[i-1].T) 
            self.B[i-1] += - self.learning_rate * self.delta[i] 
       
        pass
    
    
    def train(self, input_list, target_list):
        """
        * Train networks given data & label
        (args)
        1) input_list: list or numpy.array in shape = (# of input variables,)
        2) target_list: list or numpy.array in shape = (# of output nodes,)
           classification: target = 'one-hot', 'integer', ... encoding
           regression: target = y in 'float' encoding
        """
        self.feedforward(input_list)       
        self.backpropagate(target_list)
        
        pass

    
    def actfunc(self, a, func):

        if func=='identity':
            return a
        elif func=='sigmoid':
            return np.exp(-np.logaddexp(0, -a)) # numerically stable, avoiding exp overflow
#             return 1/(1+np.exp(-a))
        elif func=='tanh':
            return np.tanh(a)
        elif func=='softmax':
            return np.exp(a-a.max())/np.sum(np.exp(a-a.max())) # numerically stable, avoiding exp overflow
#             return np.exp(a)/np.sum(np.exp(a))
        elif func=='relu':
            return np.maximum(a,0)
        elif func=='selu': # [Self-normalizing NN](https://arxiv.org/pdf/1706.02515.pdf)
            scale=1.0507009873554804934193349852946
            alpha=1.6732632423543772848170429916717
            return scale*np.where(a<0.0, alpha*(np.exp(a)-1.0), a)
        else:
            raise ValueError(' => Check your activation function list !')

        pass

    
    def df_over_da(self,a, func):

        if func=='identity':
            return np.ones(len(a))
        elif func=='sigmoid':
            return self.actfunc(a, func)*(1-self.actfunc(a, func))
        elif func=='tanh':
            return 1-(np.tanh(a))**2
        elif func=='softmax':
            return self.actfunc(a, func)*(1-self.actfunc(a, func))
        elif func=='relu':
#            return (a>0)*1.0
            return np.where(a<0.0, 0.0, 1.0)
        elif func=='selu':
            scale=1.0507009873554804934193349852946
            alpha=1.6732632423543772848170429916717
            return scale*np.where(a<0.0, alpha*np.exp(a), 1.0)
        else:
            raise ValueError(' => Check your activation function list !')
            
        pass
    
    
    def check_a_output(self, input_list, target_list):
                    
        if self.encoding == 'one-hot':
            
            y_correct = np.argmax(target_list)
            y_predicted_raw = self.feedforward(input_list) 
            y_predicted = np.argmax(y_predicted_raw) 

            print (' * 본 클래스 = ', y_correct)
            print (' * 예측된 각 클래스에 대한 확률값 = \n', y_predicted_raw)
            print (' * 최대확률 클래스 = ', y_predicted)
        
        elif self.encoding == 'integer':
            
            y_correct = target_list[0]
            y_predicted_raw = self.feedforward(input_list) 
            y_predicted = int(y_predicted_raw)
            
            print (' * 본 클래스 = ', y_correct)
            print (' * 예측된 클래스 출력치 = \n', y_predicted_raw)
            print (' * 근접 클래스 = ', y_predicted)
            
        elif self.encoding == 'float':
             
            y_correct = target_list[0]
            y_predicted_raw = self.feedforward(input_list) 
            y_predicted = y_predicted_raw
            
            print (' * 본 라벨값 = ', y_correct)
            print (' * 신경망 출력값 = \n', y_predicted)

        return y_predicted
    
    
    def check_accuracy_error(self, data_list, id_data_min, id_data_max, data_type=None):
        """
        * 정확도와 평균값 반환 
        * 입력: ( data, id_data_min=0, id_data_max=len(data)-1, data_type=None )
        > data: numpy_array in shape of (n_samples, [label, x1, x2, ...])
        > id_data_min: 시작 데이터 아이디 (0,...)
        > id_data_max: 마지막 데이터 아이디 (...len(data)-1)
        > data_type: 데이터 유형 ('mnist' or None)
        """
        # 예측 정확도 스코어 리스트, 에러 리스트 초기화
        score_list, error_list = [], []

        # 데이터셋 순환, data in [id_data_min, id_data_max]
        for data in data_list[id_data_min:id_data_max+1]:
            
            if data_type == 'mnist':
                
                # 입력데이터의 분리 ['정답라벨', 'x1', 'x2', ...]
                all_values = data.split(',')

                # 정답라벨
                y_correct = int(all_values[0]) # (1,)
                # 정답라벨의 one-hot encoding 
                y_correct_list = np.zeros(10) #self.n_nodes[-1])
                # 정답클래스에 대응되는 요소값를 1로 변환 
                y_correct_list[int(all_values[0])] = 1.0

                # 입력데이터 ['x1','x2',...] & 입력데이터의 스케일링 (n_input_nodes, )
                input_list = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                                    
                # 예측된 클래스 확률 얻기
                y_predicted_list = self.feedforward(input_list) # (n_output_nodes, 1)
                # 최대확률의 클래스 얻기
                y_predicted = np.argmax(y_predicted_list) # (1,)

                #print (' y_correct, type    = ', y_correct,' ', type(y_correct))
                #print (' y_predicted, type  = ', y_predicted, ' ', type(y_predicted))
                
                # 정답라벨 클래스와 최대확률의 클래스의 비교
                if (y_correct == y_predicted):
                    # 같으면 스코어 리스트에 +1
                    score_list.append(1)
                else: 
                    # 틀리면 스코어 리스트에 +0
                    score_list.append(0)
                    pass

            else:

                # 입력데이터 ['x1','x2',...] 
                input_list = data[1:] #np.asfarray(all_values[1:])        
                
                # 정답라벨 데이터
                y_correct = data[0] # (1,)
       
                # 정답라벨 인코딩
                # one-hot encoded  
                if self.encoding == 'one-hot':
                    
                    y_correct_list = np.zeros(self.n_nodes[-1])
                    y_correct_list[int(y_correct)] = 1
                                   
                    # 예측된 클래스 확률 얻기
                    y_predicted_list = self.feedforward(input_list) # (n_output_nodes, 1)
                    
                    #print(' * predicted_list.shape = ', y_predicted_list)
                    
                    # 최대확률의 클래스 얻기
                    y_predicted = np.argmax(y_predicted_list) # (1,)

                    # 정답라벨 클래스와 최대확률의 클래스의 비교
                    if (y_correct == y_predicted):
                        # 같으면 스코어 리스트에 +1
                        score_list.append(1)
                    else: 
                        # 틀리면 스코어 리스트에 +0
                        score_list.append(0)
                        pass
               
                # integer encoded, y = 0, 1, 2, ...
                elif self.encoding == 'integer':
                    
                    y_correct_list = np.zeros(1)
                    y_correct_list[0] = int(y_correct)

                    # 예측된 출력값 얻기, a float around (0~N_class-1)
                    y_predicted_list = self.feedforward(input_list) # (n_output_nodes, 1) = (1,1)
                    
                    # 예측된 클래스 얻기, a rounded integer
                    y_predicted = round(y_predicted_list[0,0])
                    
                    # 정답라벨 클래스와 예측 클래스의 비교
                    if (y_correct == y_predicted):
                        # 같으면 스코어 리스트에 +1
                        score_list.append(1)
                    else: 
                        # 틀리면 스코어 리스트에 +0
                        score_list.append(0)
                        pass                   
                    
                # encoded as original value: regression
                elif self.encoding == 'float':
                    
                    y_correct_list = np.zeros(1)
                    y_correct_list[0] = y_correct
                     
                     # 예측된 출력값 얻기, a float around (0~N_class-1)
                    y_predicted_list = self.feedforward(input_list)# (n_output_nodes, 1) = (1,1)
                    y_predicted = y_predicted_list
                    
                else:
                    
                    raise ValueError(' => check your encoding scheme')
                
            
            # 오차값의 계산 (Mean Squared Error의 경우) 
            error = ((y_correct_list.reshape(len(y_correct_list),1) - y_predicted_list)**2).mean()
            # 오차값의 저장
            error_list.append(error)
            
            pass
        
        # calculate the performance score, the fraction of correct answers
        score_list_np = np.asarray(score_list)
        error_list_np = np.asarray(error_list)
        print (" ")
        if self.encoding != "float":
            print (" * 현재 정확도  = ", score_list_np.mean(), ' | (정답수)/(테스트 데이터수) = ', score_list_np.sum(),'/',score_list_np.size)
        print (" * 현재 평균에러 = ", error_list_np.mean())
        
    
    def save_model(self, fname='dnn_model.npy', nametag = None):
        """
        * saves current network model information into a big numpy array, 
          and writes the array to a file in .npy format with filename given by 'fname'
        """
        if nametag is not None:        
            model_nametag = nametag
        else:
            model_nametag = self.model_nametag
        
        W2 = self.W.copy()
        B2 = self.B.copy()
        
        save_model_np = np.r_[['*Neural Netowrk model in numpy ndarray*'],[model_nametag],\
                              [self.model_structure],[self.learning_rate],\
                              [self.encoding],\
                              self.W, self.B]
        
        np.save(fname, save_model_np)
        
        pass
    

    def load_model(self, fname='dnn_model.npy'):
    
        model_np = np.load(fname)
        self.__init__(load_model_np=model_np)
        
        pass
            
        
