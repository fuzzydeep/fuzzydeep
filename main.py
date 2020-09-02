#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import logging
import argparse
import torch

from source.inputter.corpus import KnowledgeCorpus
from source.model.MPL import MPL
from source.model.seq2seq import Seq2Seq
from source.model.seq2seq_b import Seq2Seq_B
from source.model.seq2seq_e import Seq2Seq_E
from source.utils.engine_MAML import Trainer
from source.utils.generator import BeamGenerator
from source.utils.misc import str2bool


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_name", type=str, default="camrest")
    data_arg.add_argument("--data_dir", type=str, default="./data/CamRest")
    data_arg.add_argument("--save_dir", type=str, default="./models")
    data_arg.add_argument("--embed_file", type=str, default="glove.840B.300d.txt")
    data_arg.add_argument("--pre_train_dir", type=str, default="./pre_train")
    data_arg.add_argument("--s_ckpt", type=str, default="s.model")
    data_arg.add_argument("--tb_ckpt", type=str, default="bleu.model")
    data_arg.add_argument("--te_ckpt", type=str, default="kb.model")

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=256)
    net_arg.add_argument("--bidirectional", type=str2bool, default=False)
    net_arg.add_argument("--max_vocab_size", type=int, default=30000)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=400)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--max_hop", type=int, default=3)
    net_arg.add_argument("--attn", type=str, default='mlp', choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", type=str2bool, default=True)
    net_arg.add_argument("--with_bridge", type=str2bool, default=False)
    net_arg.add_argument("--tie_embedding", type=str2bool, default=True)
    net_arg.add_argument("--meta_epochs", type=int, default=1000)
    # Training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--gpu", type=int, default=4)
    train_arg.add_argument("--batch_size", type=int, default=8)
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--lr_decay", type=float, default=0.5)
    train_arg.add_argument("--patience", type=int, default=5)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.2)
    train_arg.add_argument("--num_epochs", type=int, default=10)
    train_arg.add_argument("--pre_epochs", type=int, default=25)
    train_arg.add_argument("--log_steps", type=int, default=5)
    train_arg.add_argument("--valid_steps", type=int, default=20)
    train_arg.add_argument("--nen_weight", type=int, default=0.1)
    train_arg.add_argument("--L2", type=float, default=1e-5)
    train_arg.add_argument("--use_embed", type=str2bool, default=True)
    train_arg.add_argument("--train_embed", type=str2bool, default=True)
    # Generation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--test", default=False)
    #/home/qinbw/MGMA/models-camrest/best.model
    gen_arg.add_argument("--ckpt", type=str, default="state_epoch_13.model")
    gen_arg.add_argument("--beam_size", type=int, default=2)
    gen_arg.add_argument("--max_dec_len", type=int, default=25)
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--save_file", type=str, default="./output.txt")
    gen_arg.add_argument("--test_model", type=str, default="S", choices=['S', 'TB', 'TE'])

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()

    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    torch.cuda.set_device(device)

    # Special tokens definition
    special_tokens = ["<ENT>", "<NEN>"]
    print(config.data_dir)
    # Data definition
    corpus = KnowledgeCorpus(data_dir=config.data_dir,
                             min_freq=0, max_vocab_size=config.max_vocab_size,
                             min_len=config.min_len, max_len=config.max_len,
                             embed_file=config.embed_file, share_vocab=config.share_vocab,
                             special_tokens=special_tokens)

    corpus.load()

    # Model definition
    model_T = Seq2Seq(src_field=corpus.SRC, tgt_field=corpus.TGT,
                    kb_field=corpus.KB, embed_size=config.embed_size,
                    hidden_size=config.hidden_size, padding_idx=corpus.padding_idx,
                    num_layers=config.num_layers, bidirectional=config.bidirectional,
                    attn_mode=config.attn, with_bridge=config.with_bridge,
                    tie_embedding=config.tie_embedding, dropout=config.dropout,
                    max_hop=config.max_hop, use_gpu=config.use_gpu)
    #bleu
    model_Dialog = Seq2Seq_B(src_field=corpus.SRC, tgt_field=corpus.TGT,
                    kb_field=corpus.KB, embed_size=config.embed_size,
                    hidden_size=config.hidden_size, padding_idx=corpus.padding_idx,
                    num_layers=config.num_layers, bidirectional=config.bidirectional,
                    attn_mode=config.attn, with_bridge=config.with_bridge,
                    tie_embedding=config.tie_embedding, dropout=config.dropout,
                    max_hop=config.max_hop, use_gpu=config.use_gpu)
    #entity-F1
    model_KB =    Seq2Seq_E(src_field=corpus.SRC, tgt_field=corpus.TGT,
                         kb_field=corpus.KB, embed_size=config.embed_size,
                         hidden_size=config.hidden_size, padding_idx=corpus.padding_idx, nen_idx=corpus.nen_idx,
                         num_layers=config.num_layers, bidirectional=config.bidirectional,
                         attn_mode=config.attn, with_bridge=config.with_bridge,
                         tie_embedding=config.tie_embedding, dropout=config.dropout,
                         max_hop=config.max_hop, nen_weight=config.nen_weight, use_gpu=config.use_gpu)

    # Generator definition (note every generator only use single model generate here,
    # todo later can consider ensemble)
    generator_S = BeamGenerator(model=model_T, src_field=corpus.SRC, tgt_field=corpus.TGT,
                                kb_field=corpus.KB, beam_size=config.beam_size, max_length=config.max_dec_len,
                                ignore_unk=config.ignore_unk, length_average=config.length_average,
                                use_gpu=config.use_gpu)

    generator_TB = BeamGenerator(model=model_Dialog, src_field=corpus.SRC, tgt_field=corpus.TGT,
                                 kb_field=corpus.KB, beam_size=config.beam_size, max_length=config.max_dec_len,
                                 ignore_unk=config.ignore_unk, length_average=config.length_average,
                                 use_gpu=config.use_gpu)

    generator_TE = BeamGenerator(model=model_KB, src_field=corpus.SRC, tgt_field=corpus.TGT,
                                 kb_field=corpus.KB, beam_size=config.beam_size, max_length=config.max_dec_len,
                                 ignore_unk=config.ignore_unk, length_average=config.length_average,
                                 use_gpu=config.use_gpu)

    MAML_agent = MPL(model_KB=model_KB, model_Dialog=model_Dialog, model_T=model_T)

    # Testing (default only test model_S)
    if config.test and config.ckpt:
        test_iter = corpus.create_batches(config.batch_size, data_type="test", shuffle=False)

        model_path = os.path.join(config.save_dir, config.ckpt)
        MAML_agent.load(model_path)
        print("Testing ...")
        # if config.test_model == "S":
        #     test_model, generator = muti_agent.model_S, generator_S
        # elif config.test_model == "TB":
        #     test_model, generator = muti_agent.model_TB, generator_TB
        # elif config.test_model == "TE":
        #     test_model, generator = muti_agent.model_TE, generator_TE
        # else:
        #     print("Invaild test model and generator!")
        #     sys.exit(0)
        # metrics = Trainer.evaluate(MAML_agent.model_T, test_iter)
        # print(metrics.report_cum())
        print("Generating ...")
        generator_S.generate(data_iter=test_iter, save_file=config.save_file, verbos=True)

    else:
        train_iter = corpus.create_batches(config.batch_size, data_type="train", shuffle=True)
        valid_iter = corpus.create_batches(config.batch_size, data_type="valid", shuffle=False)

        # Optimizer definition
        optimizer_T = getattr(torch.optim, config.optimizer)(model_T.parameters(), lr=config.lr, weight_decay=config.L2)
        optimizer_KB = getattr(torch.optim, config.optimizer)(model_KB.parameters(), lr=config.lr,
                                                              weight_decay=config.L2)
        optimizer_Dialog = getattr(torch.optim, config.optimizer)(model_Dialog.parameters(), lr=config.lr,
                                                                  weight_decay=config.L2)
        optimizer = {'T': optimizer_T, 'kb': optimizer_KB, 'dialog': optimizer_Dialog}
        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler_T = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_T, mode='min', factor=config.lr_decay,
                patience=config.patience, verbose=True, min_lr=1e-6)

            lr_scheduler_KB = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_KB, mode='min', factor=config.lr_decay,
                patience=config.patience, verbose=True, min_lr=1e-6)

            lr_scheduler_Dialog = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer_Dialog, mode='min', factor=config.lr_decay,
                patience=config.patience, verbose=True, min_lr=1e-6)
            lr_scheduler = {'T': lr_scheduler_T, 'kb': lr_scheduler_KB, 'dialog': lr_scheduler_Dialog}
        else:
            lr_scheduler = None
        if config.use_embed and config.embed_file is not None:
            model_T.encoder.embedder.load_embeddings(corpus.SRC.embeddings, scale=0.03,
                                                   trainable=config.train_embed)
            model_T.decoder.embedder.load_embeddings(corpus.TGT.embeddings, scale=0.03,
                                                   trainable=config.train_embed)
            model_Dialog.encoder.embedder.load_embeddings(corpus.SRC.embeddings, scale=0.03,
                                                     trainable=config.train_embed)
            model_Dialog.decoder.embedder.load_embeddings(corpus.TGT.embeddings, scale=0.03,
                                                     trainable=config.train_embed)
            model_KB.encoder.embedder.load_embeddings(corpus.SRC.embeddings, scale=0.03,
                                                     trainable=config.train_embed)
            model_KB.decoder.embedder.load_embeddings(corpus.TGT.embeddings, scale=0.03,
                                                     trainable=config.train_embed)
        # Save directory
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        logger.info("Saved params to '{}'".format(params_file))
        logger.info(MAML_agent)

        # Training
        logger.info("Training starts ...")
        # logger.info("Learning Approach: " + config.method)
        trainer = Trainer(epochs=config.meta_epochs,model=MAML_agent, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger, valid_metric_name="-loss",
                          num_epochs=config.num_epochs, pre_epochs=config.pre_epochs,
                          save_dir=config.save_dir, pre_train_dir=config.pre_train_dir,
                          log_steps=config.log_steps, valid_steps=config.valid_steps,
                          grad_clip=config.grad_clip, lr_scheduler=lr_scheduler)

        # if config.ckpt:
        #     trainer.load(file_ckpt=config.ckpt)
        # else:
        #     # The whole pre_train model doesn't exist means we will train from scratch,
        #     # therefore load the three single pre_train model in the whole model (muti-agent)
        #     trainer.load_per_agent(S_ckpt=config.s_ckpt, TE_ckpt=config.te_ckpt, TB_ckpt=config.tb_ckpt)
        #     # close_train((model_TE, model_TB))
        # trainer.load_per_agent(S_ckpt=config.s_ckpt, TE_ckpt=config.te_ckpt, TB_ckpt=config.tb_ckpt)
        #trainer.load(file_ckpt=config.ckpt)
        trainer.train()

        logger.info("Training done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
