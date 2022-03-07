clear all; clc; close all;

% Lee imagen
I0 = imread('sky.jpg');

% Corta color de interes
[~,rect] = imcrop(I0);
close(gcf); pause(0.1);

% Normaliza canales RGB
I = double(I0)/255;
I = bsxfun(@rdivide,I,sum(I,3)+0.1);

% Region de color
J = imcrop(I,rect);

% Separa canales de la region
R = J(:,:,1);
G = J(:,:,2);
B = J(:,:,3);

% Vector de caracteristicas de entrenamiento
x = [R(:) G(:) B(:)]';
d = size(x,1); % numero de caracteristicas

% Calcula parametros del modelo probabilistico
mu = medias(x); % medias
Sigma = covarianza(x,mu); % matriz de covarianza
% Adicionalmente calcula inversa y determinante
iSigma = pinv(Sigma);
dSigma = det(Sigma);

% Evalua cada pixel de la imagen para determinar
% la probabilidad de pertenecer al color seleccionado
[n,m,k] = size(I); % Tamano de la imagen
p = zeros(n,m); % Inicializa imagen de probabilidad

for i = 1:n
    for j = 1:m
        xij = [I(i,j,1) I(i,j,2) I(i,j,3)]'; % Color del pixel
        p(i,j) = gaussfun(xij,d,mu,iSigma,dSigma);
    end
end

figure;
subplot(1,2,1); imshow(I0);
subplot(1,2,2); imshow(mat2gray(p));

%***************************************
function mu = medias(X)
    [d,n] = size(X);
    mu = zeros(d,1);
    for j = 1:n
        for k = 1:d
            mu(k) = mu(k) + X(k,j)/n;
        end
    end
end
%***************************************
function Sigma = covarianza(X,mu)
    [d,n] = size(X);
    Sigma = zeros(d,d);
    for j = 1:n
        for k = 1:d
            for l = 1:d
                Sigma(k,l) = Sigma(k,l) + (X(k,j)-mu(k))*(X(l,j)-mu(l))/(n-1);
            end
        end
    end
end
%***************************************
function p = gaussfun(x,d,mu,iSigma,dSigma)
    c1 = 1/((2*pi)^(d/2)*sqrt(dSigma));
    c2 = exp(-0.5*(x-mu)'*iSigma*(x-mu));
    p = c1*c2;
end
