from datasets.DigitFive import digit5_dataset_read
from datasets.AmazonReview import amazon_dataset_read
from datasets.OfficeCaltech10 import get_office_caltech10_dloader
from datasets.DomainNet import get_domainnet_dloader
from datasets.Office31 import get_office31_dloader

def dataloader(configs, args):
# build dataset
    train_dloaders = []
    test_dloaders = []
    if configs["DataConfig"]["dataset"] == "DigitFive":
        domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
        # [0]: target dataset, target backbone, [1:-1]: source dataset, source backbone
        # generate dataset for train and target
        print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = digit5_dataset_read(args.base_path,
                                                                        args.target_domain,
                                                                        configs["TrainingConfig"]["batch_size"])
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        # generate CNN and Classifier for target domain
        domains.remove(args.target_domain)
        args.source_domains = domains
        print("target domain {} loaded".format(args.target_domain))
        # create DigitFive dataset
        print("Source Domains :{}".format(domains))
        for domain in domains:
            # generate dataset for source domain
            source_train_dloader, source_test_dloader = digit5_dataset_read(args.base_path, domain,
                                                                            configs["TrainingConfig"]["batch_size"])
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            # generate CNN and Classifier for source domain
            print("Domain {} Preprocess Finished".format(domain))
        num_classes = 10
        return train_dloaders, test_dloaders, num_classes
    elif configs["DataConfig"]["dataset"] == "AmazonReview":
        domains = ["books", "dvd", "electronics", "kitchen"]
        print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = amazon_dataset_read(args.base_path,
                                                                        args.target_domain,
                                                                        configs["TrainingConfig"]["batch_size"])
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        # generate MLP and Classifier for target domain
        domains.remove(args.target_domain)
        args.source_domains = domains
        print("target domain {} loaded".format(args.target_domain))
        # create DigitFive dataset
        print("Source Domains :{}".format(domains))
        for domain in domains:
            # generate dataset for source domain
            source_train_dloader, source_test_dloader = amazon_dataset_read(args.base_path, domain,
                                                                            configs["TrainingConfig"]["batch_size"])
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            # generate CNN and Classifier for source domain
            print("Domain {} Preprocess Finished".format(domain))
        num_classes = 2
        return train_dloaders, test_dloaders, num_classes
    elif configs["DataConfig"]["dataset"] == "OfficeCaltech10":
        domains = ['amazon', 'webcam', 'dslr', "caltech"]
        train_dloaders = []
        test_dloaders = []
        target_train_dloader, target_test_dloader = get_office_caltech10_dloader(args.base_path,
                                                                                 args.target_domain,
                                                                                 configs["TrainingConfig"]["batch_size"]
                                                                                 , args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader, source_test_dloader = get_office_caltech10_dloader(args.base_path, domain,
                                                                                     configs["TrainingConfig"][
                                                                                         "batch_size"], args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
        num_classes = 10
        return train_dloaders, test_dloaders, num_classes
    elif configs["DataConfig"]["dataset"] == "Office31":
        domains = ['amazon', 'webcam', 'dslr']
        train_dloaders = []
        test_dloaders = []
        target_train_dloader, target_test_dloader = get_office31_dloader(args.base_path,
                                                                         args.target_domain,
                                                                         configs["TrainingConfig"]["batch_size"],
                                                                         args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader, source_test_dloader = get_office31_dloader(args.base_path, domain,
                                                                             configs["TrainingConfig"]["batch_size"],
                                                                             args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
        num_classes = 31
        return train_dloaders, test_dloaders, num_classes
    elif configs["DataConfig"]["dataset"] == "DomainNet":
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        train_dloaders = []
        test_dloaders = []
        target_train_dloader, target_test_dloader = get_domainnet_dloader(args.base_path,
                                                                          args.target_domain,
                                                                          configs["TrainingConfig"]["batch_size"],
                                                                          args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader, source_test_dloader = get_domainnet_dloader(args.base_path, domain,
                                                                              configs["TrainingConfig"]["batch_size"],
                                                                              args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
        num_classes = 345
        return train_dloaders, test_dloaders, num_classes
    else:
        raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))