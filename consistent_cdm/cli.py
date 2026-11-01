"""One command for preparing, training and applying D-only decomposition models."""
from __future__ import annotations

import argparse


def main(argv=None):
    parser=argparse.ArgumentParser(prog="decouple",description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prepare=commands.add_parser("prepare-data",help="Freeze patient split and train-only statistics")
    prepare.add_argument("--data-root",required=True)
    prepare.add_argument("--output",required=True)
    prepare.add_argument("--seed",type=int,default=1234)
    prepare.add_argument("--train-ratio",type=float,default=.9)
    prepare.add_argument("--split-file")
    prepare.add_argument("--image-size",type=int,default=256)
    prepare.add_argument("--variants",nargs="+",default=["std","aug"])
    train=commands.add_parser("train",help="Train or resume one D-only family")
    train.add_argument("--family",choices=["diffusion","flow_matching","resunet","pix2pix"],required=True)
    train.add_argument("--prepared",required=True)
    train.add_argument("--run-id",required=True)
    train.add_argument("--config")
    train.add_argument("--max-steps",type=int)
    train.add_argument("--batch-size",type=int)
    train.add_argument("--resume")
    train.add_argument("--device")
    train.add_argument("--tiny",action="store_true",help="Small network and frequent checks for smoke validation")
    inference=commands.add_parser("infer",help="Condition-only NPY or real-image inference")
    inference.add_argument("--checkpoint",required=True)
    inference.add_argument("--input",required=True)
    inference.add_argument("--output",required=True)
    inference.add_argument("--family",choices=["diffusion","flow_matching","resunet","pix2pix"])
    inference.add_argument("--config")
    inference.add_argument("--mapping",choices=["raw","quantile"])
    inference.add_argument("--calibration")
    inference.add_argument("--nose-direction",choices=["left","right"],default="right")
    inference.add_argument("--invert",action="store_true")
    inference.add_argument("--crop",nargs=4,type=int,metavar=("X0","Y0","X1","Y1"))
    inference.add_argument("--seed",type=int,default=1234)
    inference.add_argument("--device",default="cpu")
    evaluation=commands.add_parser("evaluate",help="Paired validation with a Full/2 reference")
    evaluation.add_argument("--checkpoint",required=True)
    evaluation.add_argument("--prepared",required=True)
    evaluation.add_argument("--output",required=True)
    evaluation.add_argument("--seed",type=int,default=1234)
    evaluation.add_argument("--device",default="cpu")
    stress=commands.add_parser("stress-test",help="Validation D metrics under controlled condition degradations")
    stress.add_argument("--checkpoint",required=True)
    stress.add_argument("--prepared",required=True)
    stress.add_argument("--output",required=True)
    stress.add_argument("--operators",nargs="+",help="Subset of degradations; default all")
    stress.add_argument("--mapping",choices=["raw","quantile"],default="quantile",
                        help="Condition mapping after degradation; quantile matches real-image inference")
    stress.add_argument("--max-cases",type=int,help="Use only the first N validation patients")
    stress.add_argument("--seed",type=int,default=1234)
    stress.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    from consistent_cdm import runtime
    try:
        if args.command=="prepare-data":
            result=runtime.prepare_data(args.data_root,args.output,seed=args.seed,train_ratio=args.train_ratio,
                         split_file=args.split_file,image_size=args.image_size,variants=args.variants)
            print(f"Prepared {len(result['split']['train'])} train / {len(result['split']['val'])} validation patients; s_D={result['d_normalization']['scale']:.8g}")
        elif args.command=="train": print(runtime.train(args))
        elif args.command=="infer": print(f"Saved {len(runtime.infer(args)['images'])} predictions to {args.output}")
        elif args.command=="stress-test":
            runtime.stress_test(args); print(f"Saved degradation stress test to {args.output}")
        else:
            runtime.evaluate(args); print(f"Saved paired validation to {args.output}")
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        parser.error(str(error))


if __name__=="__main__":
    main()
