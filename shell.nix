let
  system = "x86_64-linux";

  nixpie = import <nixpie>;

  inherit (nixpie.inputs.nixpkgs) lib;
  inherit (lib) attrValues;

  pkgs = import nixpie.inputs.nixpkgs {
    inherit system;
    config = {
      allowUnfree = true;
    };
    overlays = (attrValues nixpie.overlays) ++ [ nixpie.overrides.${system} ];
  };
in
pkgs.mkShell {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    git gitRepo gnupg autoconf curl
    procps gnumake utillinux m4 gperf unzip cmake
    linuxPackages.nvidia_x11
    libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib libpng pngpp tbb gbenchmark gtest
    opencv
    nlohmann_json
    ncurses5 stdenv.cc binutils
  ];
  shellHook = with pkgs;''export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
export LD_LIBRARY_PATH=${linuxPackages.nvidia_x11}/lib:${ncurses5}/lib:${libkrb5}/lib:$LD_LIBRARY_PATH
export EXTRA_LDFLAGS="-L/lib -L${linuxPackages.nvidia_x11}/lib $EXTRA_LDFLAGS"
export EXTRA_CCFLAGS="-I/usr/include $EXTRA_CCFLAGS"
'';
    }
