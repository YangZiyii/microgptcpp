#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <set>
#include <memory>
#include<cmath>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
using namespace std;

	std::mt19937 gen(42);


struct Node{
		double data;
		double grad=0.0;
		vector<shared_ptr<Node>> _children;
		vector<double> _local_grads;
		Node(double d):data(d){}
};

class Value{
	public:
		shared_ptr<Node> p;
		
		Value(double d){
			p=make_shared<Node>(d);
		}
		
		Value operator+(const Value& x) const{
			auto result=Value(p->data+x.p->data);
			result.p->_children={p,x.p};
			result.p->_local_grads={1.0,1.0};
			return result;
		}
		Value operator-(const Value& x) const{
			auto result=Value(p->data-x.p->data);
			result.p->_children={p,x.p};
			result.p->_local_grads={1.0,-1.0};
			return result;
		}
		Value operator*(const Value& x) const{
			auto result=Value(p->data*x.p->data);
			result.p->_children={p,x.p};
			result.p->_local_grads={x.p->data,p->data};
			return result;
		}

		Value operator/(const Value& x) const{
			auto result=Value(p->data/x.p->data);
			result.p->_children={p,x.p};
			result.p->_local_grads={ 1.0 / x.p->data ,- p->data /(x.p->data * x.p->data)};
			return result;
		}
		Value operator-() const{
			auto result=Value(-p->data);
			result.p->_children={p};
			result.p->_local_grads={-1.0};
			return result;
		}
		
		Value pow(const double x) const{
			auto result=Value(std::pow(p->data,x));
			result.p->_children={p,};
			result.p->_local_grads={x*std::pow(p->data,x-1)};
			return result;
		}		
		Value log() const{
			auto result=Value(std::log(p->data));
			result.p->_children={p,};
			result.p->_local_grads={1.0/p->data};
			return result;
		}
		Value relu() const{
			auto result=Value(p->data>0.0 ? p->data : 0.0);
			result.p->_children={p,};
			result.p->_local_grads={p->data>=0 ? 1.0:0.0};
			return result;
		}
		Value e() const{
			auto result=Value(exp(p->data));
			result.p->_children={p,};
			result.p->_local_grads={exp(p->data)};
			return result;
		}		
		
		void build_topo(const shared_ptr<Node>& v ,vector<shared_ptr<Node>>& topo,unordered_set<Node*>& visited){
				auto visited_position = visited.find(v.get());
				if (visited_position == visited.end()){
					visited.insert(v.get());
					for(shared_ptr<Node> child:v->_children){
						build_topo(child,topo,visited);
					}
					topo.push_back(v);
				}	
		}
				
		
		void backward(){
			vector<shared_ptr<Node>> topo;
			unordered_set<Node*> visited;
			build_topo(p,topo,visited);
			p->grad = 1;
			int total_topo = topo.size();
			for(int i = total_topo - 1;i>=0;i--){
				auto v = topo[i];
				int total_child = v->_children.size();
				for(int j=0;j<total_child;j++){
					v->_children[j]->grad = v->_children[j]->grad + v->grad * v->_local_grads[j];
				}
			}
		}
		
};

	vector<vector<Value>> matrix(int nout,int nin){
		float sigma=0.08;
		vector<vector<Value>>result(nout,vector<Value>(nin,Value(0.0)));
		normal_distribution<double> dist(0, sigma);
		for(int i=0;i<nout;i++){
			for(int j=0;j<nin;j++){
				result[i][j]=Value(dist(gen));
			}
		}
		return result;
	}
	
	vector<Value> linear(const vector<Value>& x,const vector<vector<Value>>& w){
		int in_size=x.size();
		int out_size=w.size();
		vector<Value> result(out_size, Value(0.0));
		for(int i=0;i<out_size;i++){
			for(int j=0;j<in_size;j++){
				result[i]=result[i] + x[j] * w[i][j] ;
			}
		}
		return result;
	}
	
	vector<Value> softmax(const vector<Value>& x){
		int x_size=x.size();
		Value total = Value(0.0);
		vector<Value> result(x_size, Value(0.0));
		for(int i=0;i<x_size;i++){
			total = total + x[i].e();
		}
		for(int j=0;j<x_size;j++){
			result[j] = x[j].e() / total;
		}
		return result;
	}
	vector<Value> rmsnorm(const vector<Value>& x){
		int x_size=x.size();
		Value square_total = Value(0.0);
		vector<Value> result(x_size, Value(0.0));
		for(int i=0;i<x_size;i++){
			square_total = square_total + x[i].pow(2);
		}
		Value scale = ( square_total / x_size + 0.00001 ).pow(-0.5);
		for(int j=0;j<x_size;j++){
			result[j] = x[j] * scale;	
		}
		return result;
	}
		

	vector<Value> gpt(
	int token_id,
	int pos_id,
	vector<vector<vector<Value>>>& keys,
	vector<vector<vector<Value>>>& values,
	const unordered_map<string,vector<vector<Value>>>& state_dict,
	int n_layer,
	int n_head,
	int head_dim) {
		
		const vector<Value>& tok_emb = state_dict.at("wte")[token_id]; 
    	const vector<Value>& pos_emb = state_dict.at("wpe")[pos_id];
    	
    	vector<Value> x( n_head * head_dim ,0.0);
    	for(int i = 0 ;i < n_head * head_dim ;i++){
    		x[i]=tok_emb[i]+pos_emb[i];
		}
		x = rmsnorm(x);
    	
        for(int li=0;li<n_layer;li++){
        	// 1) Multi-head Attention Block
        	vector<Value> x_residual  =  x;
        	x = rmsnorm(x);
        	
        	vector<Value> q = linear(x,state_dict.at("layer"+to_string(li)+".attn_wq"));
        	vector<Value> k = linear(x,state_dict.at("layer"+to_string(li)+".attn_wk"));
        	vector<Value> v = linear(x,state_dict.at("layer"+to_string(li)+".attn_wv"));
		
		
			keys[li].push_back(k);
			values[li].push_back(v);
		
			vector<Value> x_attn;
			for(int h=0;h < n_head;h++){
				int hs = h * head_dim;
			
				vector<Value> q_h(q.begin()+hs,q.begin()+hs+head_dim);
			
				vector<vector<Value>> k_h( pos_id + 1 ,vector<Value>(head_dim,0.0));
				for(int i = 0;i < pos_id + 1 ;i++){
					for(int j = 0;j < head_dim;j++){
						k_h[i][j] = keys[li][i][j+hs]; 
					}
				}
			
				vector<vector<Value>> v_h( pos_id + 1 ,vector<Value>(head_dim,0.0));
				for(int i = 0;i < pos_id + 1 ;i++){
					for(int j = 0;j < head_dim;j++){
						v_h[i][j] = values[li][i][j+hs]; 
					}
				}
				
			
				vector<Value> attn_logits;
				for(int i = 0;i <= pos_id;i++){
					Value single_logit = Value(0.0);
					for(int j = 0;j < head_dim ; j++){
						single_logit = single_logit + q_h[j] * k_h[i][j];
					}
					attn_logits.push_back(single_logit/sqrt(head_dim));
				}
				vector<Value> attn_weights = softmax(attn_logits);
				
				for(int i = 0;i<head_dim ; i++){
					Value single_head_out = Value(0.0);
					for(int j = 0;j<=pos_id;j++){
						single_head_out = single_head_out + attn_weights[j] * v_h[j][i];
						}
					x_attn.push_back(single_head_out);
				}
			}
			
			// 2) MLP Block
			x = linear(x_attn,state_dict.at("layer"+to_string(li)+".attn_wo"));
			for(int i = 0;i < n_head * head_dim;i++){
				x[i] = x[i] + x_residual[i];
			}
	
			x_residual  =  x;
				
			x = rmsnorm(x);
			x = linear(x,state_dict.at("layer"+to_string(li)+".mlp_fc1"));
			for(Value& xi:x){
				xi = xi.relu();
			}
			x = linear(x,state_dict.at("layer"+to_string(li)+".mlp_fc2"));
			for(int i = 0;i< n_head * head_dim ;i++){
				x[i] = x[i]+x_residual[i];	
			}
		}
		vector<Value> logits = linear(x,state_dict.at("lm_head"));
		return logits;
	}


int main(){

	vector<string> docs;
	ifstream file("input.txt");
	string line;
	while(true){
		getline(file,line);
		if(!file){
			break;
		}
		docs.push_back(line);
	}	
	std::shuffle(docs.begin(),docs.end(),gen);
	cout<<"num docs:"<<docs.size()<<endl;
	file.close();

	
	string all_chars="";
	for(string names : docs){
		all_chars=all_chars+names;
	}
	set<char> not_all_chars(all_chars.begin(),all_chars.end());
	vector<char> uchars(not_all_chars.begin(),not_all_chars.end());
	int BOS=uchars.size();
	int vocab_size=uchars.size()+1;
	cout<<"vacab size:"<<vocab_size<<endl;
	

	
	
	int n_layer = 1;
	int n_embd = 16;
	int block_size = 16;
	int n_head = 4;
	int head_dim=n_embd/n_head;
		
	unordered_map<string,vector<vector<Value>>> state_dict;
	state_dict["wte"]=matrix(vocab_size,n_embd);
	state_dict["wpe"]=matrix(block_size,n_embd);
	state_dict["lm_head"]=matrix(vocab_size,n_embd);
	
	for(int i=0;i<n_layer;i++){
		string l = "layer"+to_string(i)+".";
		state_dict[l+"attn_wq"]=matrix(n_embd,n_embd);
		state_dict[l+"attn_wk"]=matrix(n_embd,n_embd);
		state_dict[l+"attn_wv"]=matrix(n_embd,n_embd);
		state_dict[l+"attn_wo"]=matrix(n_embd,n_embd);
		state_dict[l+"mlp_fc1"]=matrix(4 * n_embd,n_embd);
		state_dict[l+"mlp_fc2"]=matrix(n_embd,4 * n_embd);
	}

	vector<shared_ptr<Node>> params;

	for (const auto& mat : state_dict) {
    	for (const auto& line : mat.second) {
        	for (const auto& v : line) {
            	params.push_back(v.p);
        	}
    	}
	}

	int len_params = params.size();
	cout<<"num params:" <<len_params<<endl;

	double learning_rate = 0.01;
	double beta1 = 0.85;
	double beta2 = 0.99;
	double eps_adam = 0.00000001;
	
	vector<double> m(len_params,0.0);
	vector<double> v(len_params,0.0);
	
	int num_steps = 1000;
	for(int step = 0;step < num_steps;step++){
		string doc = docs[step % docs.size()];
		vector<int> tokens;
		tokens.push_back(BOS);
		for(char c:doc){
			auto it = find(uchars.begin(), uchars.end(), c );
			int c_id = distance(uchars.begin(),it);
			tokens.push_back(c_id);
		}
		tokens.push_back(BOS);
		int n = (block_size < tokens.size()-1)?block_size : tokens.size()-1 ;
		vector<vector<vector<Value>>> keys(n_layer);
		vector<vector<vector<Value>>> values(n_layer);
		vector<Value> losses;
		for(int pos_id=0;pos_id < n;pos_id++){
			int token_id = tokens[pos_id];
			int target_id = tokens[pos_id + 1 ];
			vector<Value> logits = gpt(token_id, pos_id, keys, values,state_dict,n_layer,n_head,head_dim);
			vector<Value> probs = softmax(logits);
			Value loss_t = - probs[target_id].log();
			losses.push_back(loss_t);	
		}
		Value total_loss = 0.0;
		for(int i = 0;i<n;i++){
			total_loss = total_loss + losses[i];
		}
		Value loss = total_loss / n ;
		
		loss.backward();
		 
		double lr_t = learning_rate * (1.0 - double(step) / num_steps);
		
		for(int i=0 ; i< len_params;i++){
			m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
			v[i] = beta2 * v[i] + (1 - beta2) * params[i]->grad * params[i]->grad ;
			double m_hat = m[i] / (1 - pow(beta1 , (step + 1)));
        	double v_hat = v[i] / (1 - pow(beta2 , (step + 1)));
        	params[i]->data = params[i]->data - lr_t * m_hat / (pow(v_hat, 0.5 ) + eps_adam);
        	params[i]->grad = 0;
		}		
		cout<<"step"<<step<<"/"<<num_steps<<"loss"<<loss.p->data<<endl;
	}

	double temperature = 0.5;
	cout<<endl<<"--- inference (new, hallucinated names) ---"<<endl;
	for(int sample_idx = 0;sample_idx<20;sample_idx ++){
		vector<vector<vector<Value>>> keys(n_layer);
		vector<vector<vector<Value>>> values(n_layer);
		int token_id = BOS;
		string sample = "";
		for(int pos_id = 0;pos_id < block_size;pos_id++){
			vector<Value> logits = gpt(token_id, pos_id, keys, values,state_dict,n_layer,n_head,head_dim);
			
			vector<Value> probs;
			for(Value l :logits) {
				probs.push_back(l / Value(temperature));
			}
			probs = softmax(probs);
			vector<double> number_probs;
			for(Value p : probs){
				number_probs.push_back(p.p->data);
			}
			discrete_distribution<int> dist(number_probs.begin(), number_probs.end());
    		token_id = dist(gen);
			
			if(token_id == BOS){
				break;
			}
			sample = sample + uchars[token_id];
		}
		
		cout<<"sample"<<std::left << std::setw(2) << sample_idx + 1<<":"<<sample<<endl;
	}
	
	return 0;
}
