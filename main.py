# ------------------------------ BASELINE (FULL DATASET) ------------------------------ 
def main():
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1 
    # Instantiates 2 data.Field objects 
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    # Load the train, validation, and test datasets to become datasets
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='datawang/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    # Create an object that can be enumerated (for training loop later)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
    sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    
    # 3.2.4 Vocab object contains the index/token for each unique word in the dataset (looks through all sentences in dataset)
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1 Loading GloVe Vector and Using Embedding Layer
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape) #number of unique words 
    
    # 4.3 Training the baseline model --------------------
    # Reproducability 
    torch.manual_seed(seed)

    # Initiate model 
    model = Baseline(100,vocab) ### 
    
    # Define loss and optimzer functions 
    loss_fnc = nn.CrossEntropyLoss()# Convert labels to one-hot to calculate loss 
    #labels_oh = F.one_hot(labels,10)()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


    # Store for plotting
    loss_list = []
    acc_list = []
    nepoch = []

    val_acc_list = []
    val_loss_list = []

    #TRAINING LOOP --------------------
    for e in range(epochs): 
        nepoch = nepoch + [e]

        batchloss_accum = 0.0
        batchacc_accum = 0.0
        model.train() 

        # For batch in train_iter: #len train_iter is number of batches 
        for i, batch in enumerate(train_iter, 0):
            correct = 0 
            total = 0 
            
            # Zero parameter gradients
            optimizer.zero_grad()
            
            # Run model on inputs
            #print(batch.text)#(batch input = sentence, batch input length = 15s --> tuple of 2 tensors)
            batch_input, batch_input_length = batch.text

            outputs = model(batch_input)
            print(outputs.shape)
            #print(batch.label.float().shape)
            
            # Compute loss
            batchloss = loss_fnc(outputs, batch.label) #(batch.label) (tensor of 64 1s and 0s)
            batchloss_accum = batchloss_accum + batchloss.item() #added values of loss for all batches
            #print('batchloss',batchloss)
            
            batchloss.backward()
            optimizer.step()
            
            # Compute accuracy 
            batchacc = accuracy(outputs,batch.label)
            batchacc_accum = batchacc_accum + batchacc
            #print("Batch accuracy",batchacc)
            
            if i == len(train_iter)-1: #len(trainloader) is len(dataset)
                model.eval()
                vacc, vloss = evaluateBaseline(model,val_iter)
                
                
                print("avg acc/epoch", batchacc_accum/len(train_iter))
                print('[%d, %5d] avg loss/epoch: %.3f' % (e + 1, i + 1, batchloss_accum/len(train_iter)))
                print("validation loss:", vloss)
                print("validation acc:", vacc)

                loss_list = loss_list + [batchloss_accum/len(train_iter)]
                acc_list = acc_list + [batchacc_accum/len(train_iter)]
                val_acc_list.append(vacc)   
                val_loss_list.append(vloss)

                batchloss_accum = 0.0
                batchacc_accum = 0.0 
    
    # Evaluate with test dataset
    model.eval()
    tacc,tloss = evaluateBaseline(model,test_iter)
    print(tacc,tloss)

    print("Final Test Acccuracy:", tacc)
    
    #LOSS TOGETHER
    plt.plot(nepoch,loss_list, label = 'Train')
    plt.plot(nepoch,val_loss_list, label = 'Valid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title("Training vs. Validation Loss Curve for full dataset")
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show() 

    #ACCURACIES TOGETHER
    plt.plot(nepoch,acc_list, label = 'Train')
    plt.plot(nepoch,val_acc_list, label = 'Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy") 
    plt.title("Training vs. Validation Accuracy Curve for full dataset")
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show() 
    
    #torch.save(model,'models/model_baseline.pt')


if __name__ == '__main__':
    main()



