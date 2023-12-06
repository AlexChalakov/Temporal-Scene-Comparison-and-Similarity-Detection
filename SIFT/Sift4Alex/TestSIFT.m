function TestSIFT()

SIFTDir = 'vlfeat-0.9.21/vlfeat-0.9.21/toolbox/';
run([SIFTDir 'vl_setup']);

fol = "sift_output";
mkdir(fol);

%imfolder = 'D:\TMPYu\TestWithSIFT\google_earth\slightly_positions\1024_768\';
%im1 = imrotate(imread([imfolder 'fixed_2020_815.jpg']),0,'crop');
%im2 = imread([imfolder 'moving_size_change20.jpg']);
%im1 = imread('im1.jpg');
%im2 = imread('im2.jpg');
tab = readtable("sample_submission_Medium.csv");
for i = 1:height(tab)
    im = tab.img_pair{i,1};
    imf = strfind(im,"-");
    im1f = imf(1:imf(1)-1);
    im2f = imf(imf(1)+1:end);
    im1 = imread(im1f);
    im2 = imread(im2f);

    [mosaic,dh1,dh2,f1,f2,matches,ok,numMatches,im1_,im2_] = sift_mosaic(im1, im2);
    %mosaic = sift_mosaic();

    smatches = matches(:,ok);

    figure(10);
    imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
    o = size(im1,2) ;
    line([f1(1,smatches(1,1));f2(1,smatches(2,1))+o], ...
        [f1(2,smatches(1,1));f2(2,smatches(2,1))]) ;
    title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
        sum(ok), ...
        100*sum(ok)/numMatches, ...
        numMatches)) ;
    axis image off ;
    saveas(gcf,[fol '/im_' numstr(i) 'fig10.png'])

    smatches = matches(:,ok);

    keypoints_im1 = [f1(1,smatches(1,:));f1(2,smatches(1,:))]; % keypoints for image 1
    keypoints_im2 = [f2(1,smatches(2,:));f2(2,smatches(2,:))]; % keypoints for image 2

    figure(10);
    imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
    o = size(im1,2) ;
    hold on
    plot(keypoints_im1(1,:),keypoints_im1(2,:),'rx')
    plot(keypoints_im2(1,:)+o,keypoints_im2(2,:),'rx')
    hold off
    line([f1(1,smatches(1,1:10));f2(1,smatches(2,1:10))+o], ...
        [f1(2,smatches(1,1:10));f2(2,smatches(2,1:10))],'LineWidth',2) ;
    title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
        sum(ok), ...
        100*sum(ok)/numMatches, ...
        numMatches)) ;
    axis image off ;
    saveas(gcf,[fol '/im_' numstr(i) 'fig10_2.png'])

    figure(31),imshowpair(im1,im2);
    saveas(gcf,[fol '/im_' numstr(i) 'fig31.png'])
    figure(32),imshowpair(im1_,im2_);
    saveas(gcf,[fol '/im_' numstr(i) 'fig32.png'])
end
