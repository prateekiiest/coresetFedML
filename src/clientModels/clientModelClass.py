import copy
from torch.autograd import Variable
from src.model import *
from src.clientModels.clientBaseClass import Client
import numpy as np

torch.autograd.set_detect_anomaly(True)
class ClientModelClass(Client):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, personal_learning_rate, device, output_dim=10):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, device, output_dim=output_dim)

        self.output_dim = output_dim
        self.batch_size = batch_size
        self.N_Batch = len(train_data) // batch_size
        self.personal_learning_rate = personal_learning_rate
        self.optimizer1 = torch.optim.Adam(
            self.personal_model.parameters(), lr=self.personal_learning_rate)
        self.optimizer11 = torch.optim.Adam(
            self.personal_model.parameters(), lr=self.personal_learning_rate)
        self.optimizer2 = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.optimizer3 = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.coreset_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        N_Samples = 1
        Round = 5
        self.model.train()
        self.personal_model.train()

        # for epoch in range(1, corset_epochs  + 1):
        # initialize some weight vector w
        # w to be used for training
        true_w = np.zeros([100,1])
        true_supp = np.random.permutation(100)[:50]
        true_w[true_supp] = np.random.randn(50,1)
        true_w = torch.tensor(true_w, dtype=torch.float32, requires_grad=True)
        
        for epoch_final in range(1,  100):
            for epoch in range(1, self.local_epochs + 1):

                X, Y = self.get_next_train_batch()

                print(X.shape)

                batch_X = Variable(X.view(self.batch_size, -1))
                
                #print("Batch X", batch_X)
                
                #print("\n")
                

                
                batch_X_coreset = true_w.T@(batch_X)
                #X_coreset = 
                #print("Batch X Coreset", batch_X_coreset)
                
                #print("\n")                
                #print("===================\n")
                #print("Batch coreset", batch_X_coreset)
                #print("===================\n")
                #print("Batch coreset shape", true_w.shape)

                #batch_X_coreset = Variable(batch_X_coreset.view(self.batch_size, -1))
                batch_Y = Variable(Y.view(self.batch_size, -1))
                label_one_hot = F.one_hot(
                    batch_Y, num_classes=self.output_dim).squeeze(dim=1)

                for r in range(1, Round + 1):
                    # personal model
                    epsilons = self.personal_model.sample_epsilons(
                        self.model.layer_param_shapes)
                    layer_params1 = self.personal_model.transform_gaussian_samples(
                        self.personal_model.mus, self.personal_model.rhos, epsilons)

                    personal_output = self.personal_model.net(
                        batch_X, layer_params1)
                    # calculate the loss
                    personal_loss = self.personal_model.combined_loss_personal(
                        personal_output, label_one_hot, layer_params1,
                        self.personal_model.mus, self.personal_model.sigmas,
                        copy.deepcopy(self.model.mus),
                        [t.clone().detach() for t in self.model.sigmas], self.local_epochs)

                    epsilons_coreset = self.personal_model.sample_epsilons(
                        self.model.layer_param_shapes)
                    layer_params1_coreset = self.personal_model.transform_gaussian_samples_coreset(
                        self.personal_model.coreset_mus, self.personal_model.coreset_rhos, epsilons_coreset)

                    personal_output_coreset = self.personal_model.net(
                        batch_X_coreset, layer_params1_coreset)

                    personal_loss_coreset = self.personal_model.combined_loss_personal_coreset(
                        personal_output_coreset, label_one_hot, layer_params1_coreset,
                        self.personal_model.coreset_mus, self.personal_model.coreset_sigmas,
                        copy.deepcopy(self.model.mus),
                        [t.clone().detach()
                        for t in self.model.sigmas], self.local_epochs
                    )

                    self.optimizer1.zero_grad()
                    personal_loss.backward(retain_graph=True)
                    self.optimizer1.step()

                    self.optimizer11.zero_grad()
                    personal_loss_coreset.backward(retain_graph=True)

                    self.optimizer11.step()

                # local model
                epsilons = self.model.sample_epsilons(
                    self.model.layer_param_shapes)
                layer_params2 = self.model.transform_gaussian_samples(
                    self.model.mus, self.model.rhos, epsilons)

                layer_params2_coreset = self.model.transform_gaussian_samples(
                    self.model.coreset_mus, self.model.coreset_rhos, epsilons)

                model_output = self.model.net(batch_X, layer_params2)

                model_output_coreset = self.model.net(
                    batch_X_coreset, layer_params2_coreset)
                # calculate the loss
                model_loss = self.model.combined_loss_local(
                    [t.clone().detach() for t in layer_params1],
                    copy.deepcopy(self.personal_model.mus),
                    [t.clone().detach() for t in self.personal_model.sigmas],
                    self.model.mus, self.model.sigmas, self.local_epochs)

                model_loss_coreset = self.model.combined_loss_local_coreset(
                    [t.clone().detach() for t in layer_params1_coreset],
                    copy.deepcopy(self.personal_model.coreset_mus),
                    [t.clone().detach()
                    for t in self.personal_model.coreset_sigmas],
                    self.model.mus, self.model.sigmas, self.local_epochs)

                self.optimizer2.zero_grad()
                model_loss.backward(retain_graph=True)

                self.optimizer3.zero_grad()

                model_loss_coreset.backward(retain_graph=True)
                self.optimizer2.step()

                self.optimizer3.step()

            ###########

            # at the end of for loop we will have the optimal loss for both
            # the coreset loss and original loss. For each client will have a weight vector
            
            print("\n")
          
            print("\n")

            #K_L_q_q_w = torch.sum(kl_divergence(Normal([t.clone().detach()
             #       for t in self.personal_model.mus], [t.clone().detach()
              #      for t in self.personal_model.sigmas]),
               #                                     Normal([t.clone().detach()
                #    for t in self.personal_model.coreset_mus], [t.clone().detach()
                 #   for t in self.personal_model.coreset_sigmas]))) 

            K_L_q_q_w = sum([torch.sum(kl_divergence(Normal(self.personal_model.mus[i].clone(), self.personal_model.sigmas[i].clone()),
                                                            Normal(self.personal_model.coreset_mus[i].clone(), self.personal_model.coreset_sigmas[i].clone()))) for i in range(len(copy.deepcopy(self.personal_model.mus)))])
            # K_L_q_q_w.backward()
            
            print(K_L_q_q_w)
            self.coreset_optimizer.zero_grad()
            K_L_q_q_w.backward(retain_graph=True)

            self.coreset_optimizer.step()
            ###########

        return LOSS
